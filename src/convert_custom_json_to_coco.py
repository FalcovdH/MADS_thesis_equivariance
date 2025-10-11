
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image

def load_custom(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_mode(x: float, y: float, w: float, h: float, ow: int, oh: int) -> str:
    """
    Return 'px' if values look like pixels,
           'pct' if 0–100,
           'norm' if 0–1.
    Preference order: if anything > 100 it's pixels; elif <=1 -> norm; elif <=100 -> pct.
    """
    vals = [x, y, w, h]
    if any(v > 100 for v in vals):
        return "px"
    if all(0.0 <= v <= 1.0 for v in vals):
        return "norm"
    if all(0.0 <= v <= 100.0 for v in vals):
        return "pct"
    # fallback heuristic: if width/height <= 1 but x/y > 1 and <=100 -> mixed, assume pct
    return "pct"

def to_pixels(x: float, y: float, w: float, h: float, ow: int, oh: int) -> Tuple[float, float, float, float]:
    mode = detect_mode(x, y, w, h, ow, oh)
    if mode == "px":
        return x, y, w, h
    elif mode == "pct":
        return x * ow / 100.0, y * oh / 100.0, w * ow / 100.0, h * oh / 100.0
    else:  # norm
        return x * ow, y * oh, w * ow, h * oh

def unique_categories(data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    names = []
    seen = set()
    for fname, items in data.items():
        for ann in items:
            labels = ann.get("rectanglelabels") or ann.get("labels") or []
            for lab in labels:
                if lab not in seen:
                    seen.add(lab)
                    names.append(lab)
    if not names:
        names = ["object"]
    return names

def find_image_size(images_dir: Path, filename: str, fallback_w: int = None, fallback_h: int = None) -> Tuple[int, int]:
    p = images_dir / filename
    if p.exists():
        try:
            with Image.open(p) as im:
                return int(im.width), int(im.height)
        except Exception:
            pass
    if fallback_w and fallback_h:
        return int(fallback_w), int(fallback_h)
    # last resort: raise
    raise FileNotFoundError(f"Cannot determine size for image '{filename}'. File missing and no fallback dims provided.")

def build_coco(
    images_dir: Path,
    data: Dict[str, List[Dict[str, Any]]],
    include_unlabeled: bool
) -> Dict[str, Any]:
    # categories
    cat_names = unique_categories(data)
    categories = [{"id": i+1, "name": name} for i, name in enumerate(cat_names)]
    cat_to_id = {c["name"]: c["id"] for c in categories}

    # images
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    # Start with labeled images from JSON
    for filename, items in data.items():
        # Prefer original dims from any item; fallback to reading the image
        ow = items[0].get("original_width") if items else None
        oh = items[0].get("original_height") if items else None
        w, h = find_image_size(images_dir, filename, fallback_w=ow, fallback_h=oh)

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        for ann in items:
            labels = ann.get("rectanglelabels") or ann.get("labels") or []
            if not labels:
                continue
            cat_name = labels[0]
            cat_id = cat_to_id.get(cat_name, 1)

            x = float(ann["x"])
            y = float(ann["y"])
            bw = float(ann["width"])
            bh = float(ann["height"])

            px, py, pw, ph = to_pixels(x, y, bw, bh, w, h)

            # clamp to image bounds
            px = max(0.0, min(px, w - 1e-6))
            py = max(0.0, min(py, h - 1e-6))
            pw = max(0.0, min(pw, w - px))
            ph = max(0.0, min(ph, h - py))

            if pw <= 0 or ph <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(px, 2), round(py, 2), round(pw, 2), round(ph, 2)],
                "area": round(pw * ph, 2),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    # Optionally include unlabeled images from the folder
    if include_unlabeled:
        labeled = set(data.keys())
        for p in images_dir.iterdir():
            if not p.is_file():
                continue
            if p.name in labeled:
                continue
            try:
                with Image.open(p) as im:
                    w, h = int(im.width), int(im.height)
            except Exception:
                continue
            images.append({
                "id": img_id,
                "file_name": p.name,
                "width": w,
                "height": h
            })
            # no annotations
            img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def main():
    # === handmatige instellingen hier ===
    images_dir = Path(r"C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\raw\images")
    in_json = Path(r"C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\raw\labels.json")
    out_json = Path(r"C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\raw\annotations_coco.json")
    include_unlabeled = True  # zet op False als je alleen gelabelde images wilt

    # === logica blijft gelijk ===
    data = load_custom(in_json)
    coco = build_coco(images_dir, data, include_unlabeled=include_unlabeled)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote COCO JSON -> {out_json}")
    print(f" Images: {len(coco['images'])}")
    print(f" Annotations: {len(coco['annotations'])}")
    print(f" Categories: {[c['name'] for c in coco['categories']]}")


if __name__ == "__main__":
    main()
