
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_tile_name(name: str):
    """
    Herkent o.a.:
      base_x<X>_y<Y>_w<W>_h<H>.ext
      base__x<X>__y<Y>__w<W>__h<H>.ext
      base__sahi__x_<X>_y_<Y>_w_<W>_h_<H>.ext
      base_<n1>_<n2>_<n3>_<n4>[_<n5>].ext  (zoals: ff4..._843_2048_0_3328_1280.png)
    Retourneert (base, x, y, w, h) waar mogelijk; anders (base, None, None, None, None).
    """
    from pathlib import Path
    import re
    stem = Path(name).stem

    # 1) Gelabelde patronen
    m = re.search(r"^(?P<base>.+)_x(?P<x>\d+)_y(?P<y>\d+)_w(?P<w>\d+)_h(?P<h>\d+)$", stem)
    if not m:
        m = re.search(r"^(?P<base>.+)__x(?P<x>\d+)__y(?P<y>\d+)__w(?P<w>\d+)__h(?P<h>\d+)$", stem)
    if not m:
        m = re.search(r"^(?P<base>.+)__sahi__.*?x[_-](?P<x>\d+).*?y[_-](?P<y>\d+).*?w[_-](?P<w>\d+).*?h[_-](?P<h>\d+)$", stem)
    if m:
        d = m.groupdict()
        return d["base"], int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])

    # 2) Ongeëtiketteerde suffix met 4-5 integers aan het eind, gescheiden door underscores
    parts = stem.split("_")
    suffix_ints = []
    for p in reversed(parts):
        if p.isdigit():
            suffix_ints.append(int(p))
        else:
            break
    suffix_ints = list(reversed(suffix_ints))  # in volgorde van links->rechts

    if len(suffix_ints) >= 4:
        # Base is alles vóór die suffix
        base_len = len(parts) - len(suffix_ints)
        base = "_".join(parts[:base_len]).rstrip("_-")

        # Heuristiek: neem de laatste 4 als x,y,w,h. (Veelvoorkomend.)
        # Als je later wil, kun je deze mapping verifiëren met de tile-dims uit COCO en zo nodig omwisselen.
        if len(suffix_ints) >= 4:
            x, y, w, h = suffix_ints[-4], suffix_ints[-3], suffix_ints[-2], suffix_ints[-1]
            return base if base else None, x, y, w, h

    # 3) Niet te parsen → tenminste base teruggeven
    return stem, None, None, None, None


def load_coco(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_annotations(split_dir: Path) -> Path:
    candidates = [
        "annotations.json",
        "annotations.coco.json",
        "annotations.json_coco.json",
        "instances_default.json",
        "combined_coco.json",
    ]
    for name in candidates:
        p = split_dir / name
        if p.exists():
            return p
    # last resort: pick the first json file
    jsons = list(split_dir.glob("*.json"))
    if jsons:
        return jsons[0]
    raise FileNotFoundError(f"No annotations JSON found in {split_dir}. Tried: {', '.join(candidates)}")

def count_images_in(dir_path: Path) -> int:
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

def find_images_dir(split_dir: Path) -> Path:
    # Preferred: split/images
    p = split_dir / "images"
    if p.exists() and p.is_dir() and count_images_in(p) > 0:
        return p
    # If images directly in split_dir
    if count_images_in(split_dir) > 0:
        return split_dir
    # Otherwise: look for first subfolder with images
    for sub in split_dir.iterdir():
        if sub.is_dir() and count_images_in(sub) > 0:
            return sub
    raise FileNotFoundError(f"No images folder found under {split_dir}. Expected 'images' or image files in split.")

def check_split(split_dir: Path, tile_glob: str = "*", previews: int = 0, seed: int = 42, max_tiles: int = 0) -> Dict[str, Any]:
    ann_path = find_annotations(split_dir)
    img_dir = find_images_dir(split_dir)

    coco = load_coco(ann_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in categories}

    # Filter images by filename glob
    from fnmatch import fnmatch
    images = [im for im in images if fnmatch(im.get("file_name", ""), tile_glob)]

    # Optional sampling of a subset for speed
    if max_tiles and max_tiles > 0 and len(images) > max_tiles:
        rnd = random.Random(seed)
        images = rnd.sample(images, max_tiles)

    id_allowed = {im["id"] for im in images}
    annotations = [a for a in annotations if a["image_id"] in id_allowed]

    img_by_id = {im["id"]: im for im in images}
    anns_by_img = defaultdict(list)
    for a in annotations:
        anns_by_img[a["image_id"]].append(a)

    total_images = len(images)
    total_annotations = len(annotations)

    # BBox validity and class counts
    oob = []
    cat_counts = Counter()
    for a in annotations:
        im = img_by_id.get(a["image_id"])
        if not im:
            oob.append((a["id"], a["image_id"], "missing image"))
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > im["width"] + 1e-6 or y + h > im["height"] + 1e-6:
            oob.append((a["id"], a["image_id"], a["bbox"], (im["width"], im["height"])))
        cat_counts[a["category_id"]] += 1

    # Origin detection from filenames
    origins = set()
    parse_fail = 0
    for im in images:
        base, *_ = parse_tile_name(im["file_name"])
        if base is None:
            parse_fail += 1
        else:
            origins.add(base)

    # Optional previews
    previews_written = 0
    if previews > 0:
        rnd = random.Random(seed)
        candidate_imgs = [im for im in images if anns_by_img.get(im["id"])]
        rnd.shuffle(candidate_imgs)
        previews_dir = split_dir / "_previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        for im in candidate_imgs[:previews]:
            img_path = img_dir / im["file_name"]
            if not img_path.exists():
                continue
            try:
                im_pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            draw = ImageDraw.Draw(im_pil)
            for a in anns_by_img[im["id"]]:
                x, y, w, h = a["bbox"]
                x1, y1 = x + w, y + h
                draw.rectangle([x, y, x1, y1], outline=255, width=2)
                catname = cat_id_to_name.get(a["category_id"], str(a["category_id"]))
                try:
                    draw.text((x + 3, y + 3), catname)
                except Exception:
                    pass
            outp = previews_dir / f"preview_{im['file_name']}"
            try:
                im_pil.save(outp)
                previews_written += 1
            except Exception:
                pass

    report = {
        "split": split_dir.name,
        "images": total_images,
        "annotations": total_annotations,
        "categories": len(categories),
        "cat_distribution": {cat_id_to_name.get(k, str(k)): v for k, v in sorted(cat_counts.items())},
        "oob_boxes": len(oob),
        "parse_failures": parse_fail,
        "unique_origins_detected": len(origins),
        "previews_written": previews_written,
        "annotations_json": str(ann_path),
        "images_dir": str(img_dir),
    }
    return report

def write_manifest(split_dir: Path, tile_glob: str = "*") -> Path:
    import csv
    from fnmatch import fnmatch
    ann_path = find_annotations(split_dir)
    coco = load_coco(ann_path)
    images = [im for im in coco["images"] if fnmatch(im.get("file_name", ""), tile_glob)]
    id_allowed = {im["id"] for im in images}
    anns = [a for a in coco["annotations"] if a["image_id"] in id_allowed]

    anns_by_img = defaultdict(int)
    for a in anns:
        anns_by_img[a["image_id"]] += 1

    rows = []
    for img in images:
        file_name = img["file_name"]
        base, x, y, w, h = parse_tile_name(file_name)
        rows.append({
            "split": split_dir.name,
            "tile_file": file_name,
            "tile_w": img["width"],
            "tile_h": img["height"],
            "ann_count": anns_by_img.get(img["id"], 0),
            "origin_base": base or "",
            "tile_x0": x or "",
            "tile_y0": y or "",
            "tile_w_ref": w or "",
            "tile_h_ref": h or "",
        })

    manifest_dir = split_dir.parent / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{split_dir.name}_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    return manifest_path

def main():
    ap = argparse.ArgumentParser(description="Verify SAHI sliced dataset with auto-detected layout; selective checks; optional manifests and previews.")
    ap.add_argument("--root", type=str, required=True, help="Root containing split directories (e.g., 'train', 'val').")
    ap.add_argument("--only_split", choices=["train", "val", "both"], default="both", help="Which split(s) to check.")
    ap.add_argument("--tile_glob", type=str, default="*", help="Glob to filter tile filenames, e.g. '*_x0_*' or 'IMG_2024*'.")
    ap.add_argument("--sample_overlays", type=int, default=0, help="How many preview tiles to render per split (with boxes).")
    ap.add_argument("--skip_manifests", action="store_true", help="Do not write CSV manifests (faster).")
    ap.add_argument("--max_tiles", type=int, default=0, help="Max number of tiles to check per split (0=all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling previews and subsets.")
    args = ap.parse_args()

    root = Path(args.root)
    # Consider any immediate subdirectory as a split (train/val/test...)
    splits: List[Path] = []
    if args.only_split == "both":
        for sub in ["train", "val"]:
            p = root / sub
            if p.exists():
                splits.append(p)
    else:
        p = root / args.only_split
        if p.exists():
            splits.append(p)

    if not splits:
        raise SystemExit(f"No split directories found to check under {root}. Looked for 'train' and/or 'val'.")

    reports = []
    manifests = []
    for split in splits:
        rep = check_split(split, tile_glob=args.tile_glob, previews=args.sample_overlays, seed=args.seed, max_tiles=args.max_tiles)
        reports.append(rep)
        if not args.skip_manifests:
            mpath = write_manifest(split, tile_glob=args.tile_glob)
            manifests.append(mpath)

    print("\n=== Verification Report ===")
    for rep in reports:
        print(f"[{rep['split']}] images={rep['images']}  annotations={rep['annotations']}  categories={rep['categories']}")
        print(f"           oob_boxes={rep['oob_boxes']}  parse_failures={rep['parse_failures']}  unique_origins={rep['unique_origins_detected']}  previews={rep['previews_written']}")
        print(f"           annotations_json: {rep['annotations_json']}")
        print(f"           images_dir      : {rep['images_dir']}")
        if rep["cat_distribution"]:
            cats = ', '.join([f"{k}:{v}" for k, v in rep["cat_distribution"].items()])
            print(f"           class_dist     : {cats}")

    if manifests:
        print("\nManifests:")
        for m in manifests:
            print(" -", m)

if __name__ == "__main__":
    main()
