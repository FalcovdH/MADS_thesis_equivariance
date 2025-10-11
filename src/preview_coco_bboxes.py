# src/preview_coco_bboxes.py
import argparse, json, random
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont

def load_coco(ann_path: Path):
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id2img = {im["id"]: im for im in data["images"]}
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in data.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)
    return id2img, id2name, anns_by_img

def resolve_image_path(root: Path, split: str, file_name: str, ann_is_combined: bool) -> Path:
    """Zoek het juiste pad naar de image. Werkt voor combined en split-jsons."""
    # Als combined json: file_name kan 'train/...' of 'val/...' bevatten
    p = root / file_name if ann_is_combined else root / split / "images" / file_name
    if p.exists():
        return p
    # fallback: soms staan ze direct in de split-map
    q = (root / split / file_name)
    if q.exists():
        return q
    raise FileNotFoundError(f"Image not found for file_name={file_name} (tried: {p} and {q})")

def draw_boxes(img: Image.Image, anns: List[Dict[str, Any]], id2name: Dict[int, str]) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for a in anns:
        x, y, w, h = a["bbox"]
        x2, y2 = x + w, y + h
        # semi-transparant overlay + rand
        draw.rectangle([x, y, x2, y2], outline=(0, 255, 0, 255), width=2)
        draw.rectangle([x, y, x2, y2], fill=(0, 255, 0, 40))
        cls = id2name.get(a["category_id"], str(a["category_id"]))
        label = f"{cls}"
        tw, th = draw.textlength(label, font=font), (font.size if font else 10)
        # label background
        draw.rectangle([x, max(0, y - (th + 4)), x + tw + 6, y], fill=(0, 128, 0, 200))
        # text
        draw.text((x + 3, max(0, y - (th + 3))), label, fill=(255, 255, 255, 255), font=font)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root van dataset (bevat train/, val/, annotations.json)")
    ap.add_argument("--ann", type=str, default="annotations.json",
                    help="Pad naar COCO json. Standaard de gecombineerde JSON in de root.")
    ap.add_argument("--split", choices=["train","val","all"], default="all",
                    help="Alleen train, val of all (alle images). Bij combined-json maakt dit niet uit.")
    ap.add_argument("--n", type=int, default=5, help="Aantal previews")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    ann_path = (root / args.ann) if not Path(args.ann).is_absolute() else Path(args.ann)
    if not ann_path.exists():
        # val terug naar split-jsons
        if args.split == "all":
            raise FileNotFoundError(f"COCO json niet gevonden: {ann_path}. Probeer --ann train/annotations.json of val/annotations.json")
    id2img, id2name, anns_by_img = load_coco(ann_path)

    # Bepaal of dit een combined-json is (file_name bevat dan prefix 'train/' of 'val/')
    # Heuristiek: als een file_name een pad-separator bevat, aannemen dat het combined is
    ann_is_combined = any(('/' in im["file_name"] or '\\' in im["file_name"]) for im in id2img.values())

    # Filter optioneel op split
    img_items = list(id2img.items())
    if args.split != "all" and not ann_is_combined:
        # split-json (train of val): alle images in dit json zijn al die split
        pass
    elif args.split != "all" and ann_is_combined:
        # combined: filter op prefix in file_name
        pref = f"{args.split}/"
        img_items = [(iid, im) for iid, im in img_items if im["file_name"].replace("\\","/").startswith(pref)]

    # sample
    random.seed(args.seed)
    random.shuffle(img_items)
    sel = img_items[:max(1, args.n)]

    out_dir = root / "_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    for iid, im in sel:
        file_name = im["file_name"]
        split_guess = "train" if file_name.replace("\\","/").startswith("train/") else ("val" if file_name.replace("\\","/").startswith("val/") else args.split)
        img_path = resolve_image_path(root, split_guess, file_name, ann_is_combined)

        img = Image.open(img_path).convert("RGB")
        anns = anns_by_img.get(iid, [])
        img = draw_boxes(img, anns, id2name)

        out_name = f"{Path(file_name).stem}_preview.png"
        img.save(out_dir / out_name)

    print(f"[DONE] Saved {len(sel)} previews to: {out_dir}")

if __name__ == "__main__":
    main()
