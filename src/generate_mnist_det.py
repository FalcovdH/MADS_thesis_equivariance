# src/generate_mnist_coco_det.py
# -----------------------------------------------------------------------------
# Maak een synthetische MNIST-detectiedataset met COCO-annotaties.
# - Per split:  train/images/*.png + train/annotations.json
#               val/images/*.png   + val/annotations.json
# - Gecombineerd: annotations.json in de root (file_name met 'train/' of 'val/' prefix)
#
# Bbox-formaat: [x, y, w, h]  (COCO)
# Segmentation: rechthoek (polygon met 4 punten, 8 floats)
# Categories  : id=1..10, name="0".."9"
#
# Voorbeeld:
#   python -u src/generate_mnist_coco_det.py --out "C:/tmp/mnist_det" \
#     --imgsz 128 --n_train 200 --n_val 50 \
#     --min_digits 1 --max_digits 3 \
#     --size_mode pixels --digit_min_px 18 --digit_max_px 28 --max_rot 25
# -----------------------------------------------------------------------------

import argparse, json, random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

# ---------- helpers ----------

def coco_rect_segmentation(x: float, y: float, w: float, h: float) -> List[List[float]]:
    """Rechthoekige polygon-segmentation (x1,y1, x1,y2, x2,y2, x2,y1)."""
    return [[
        float(x),       float(y),
        float(x),       float(y + h),
        float(x + w),   float(y + h),
        float(x + w),   float(y),
    ]]

def make_rgba_digit(src_gray: Image.Image, size: int, rot_deg: float) -> Image.Image:
    """
    MNIST 'L' (0..255) -> RGBA (wit op transparant), rescale en rotate.
    Alpha = intensiteit; rotatie met expand=True houdt bbox correct.
    """
    img = src_gray.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)                 # (H,W)
    rgb = np.stack([arr]*3, axis=-1)                    # (H,W,3)
    alpha = arr                                         # (H,W)
    rgba = np.concatenate([rgb, alpha[..., None]], -1)  # (H,W,4)
    paste = Image.fromarray(rgba, mode="RGBA")
    if abs(rot_deg) > 1e-6:
        paste = paste.rotate(rot_deg, resample=Image.BILINEAR, expand=True)
    return paste

def paste_rgba(canvas_rgb: Image.Image, digit_rgba: Image.Image, x: int, y: int) -> Tuple[int,int]:
    """Plak RGBA-digit op RGB-canvas op (x,y). Return (w,h)."""
    canvas_rgb.paste(digit_rgba, (x, y), digit_rgba)
    return digit_rgba.size

def make_coco(images: List[Dict[str,Any]],
              annotations: List[Dict[str,Any]],
              categories: List[Dict[str,Any]]) -> Dict[str,Any]:
    return {"images": images, "annotations": annotations, "categories": categories}

# ---------- core ----------

def build_split(
    out_dir: Path,
    n_images: int,
    img_size: int,
    mnist_imgs: np.ndarray,
    mnist_labels: np.ndarray,
    digits_min: int,
    digits_max: int,
    rng: random.Random,
    size_mode: str,
    min_scale: float,
    max_scale: float,
    digit_min_px: int,
    digit_max_px: int,
    max_rot: float,
    start_img_id: int,
    start_ann_id: int,
) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], int, int]:
    """
    Genereer één split. Return (images, annotations, next_img_id, next_ann_id).
    """
    W = H = int(img_size)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str,Any]] = []
    annotations: List[Dict[str,Any]] = []
    img_id = start_img_id
    ann_id = start_ann_id

    for _ in range(n_images):
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        n_digits = rng.randint(digits_min, digits_max)

        anns_this_img: List[Dict[str,Any]] = []
        for _ in range(n_digits):
            idx = rng.randrange(len(mnist_imgs))
            label = int(mnist_labels[idx])
            src = Image.fromarray(mnist_imgs[idx], mode="L")

            # Groottekeuze
            if size_mode == "fraction":
                base = min(W, H)
                size = max(6, int(base * rng.uniform(min_scale, max_scale)))
            else:
                size = max(6, min(int(rng.randint(digit_min_px, digit_max_px)), min(W, H)))

            rot = rng.uniform(-max_rot, max_rot)
            paste = make_rgba_digit(src, size=size, rot_deg=rot)
            pw, ph = paste.size

            # Zorg dat hij binnen het canvas past (eventueel klein beetje rescalen)
            if pw >= W or ph >= H:
                scale = 0.9 * min(W/pw, H/ph)
                paste = paste.resize((max(1, int(pw * scale)), max(1, int(ph * scale))), Image.BILINEAR)
                pw, ph = paste.size

            x = rng.randint(0, max(0, W - pw))
            y = rng.randint(0, max(0, H - ph))
            pw, ph = paste_rgba(canvas, paste, x, y)

            bbox = [float(x), float(y), float(pw), float(ph)]
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": label + 1,  # COCO ids 1..10
                "bbox": bbox,
                "area": float(pw * ph),
                "iscrowd": 0,
                "segmentation": coco_rect_segmentation(x, y, pw, ph),
            }
            anns_this_img.append(ann)
            ann_id += 1

        fname = f"mnistdet_{img_id:06d}.png"
        (images_dir / fname).write_bytes(canvas.tobytes() if False else b"")  # avoid linter
        canvas.save(images_dir / fname)

        images.append({
            "id": img_id,
            "file_name": fname,  # prefix wordt pas bij merge gezet
            "width": W,
            "height": H
        })
        annotations.extend(anns_this_img)
        img_id += 1

    print(f"[OK] {out_dir.name}: images={len(images)}, anns={len(annotations)}")
    return images, annotations, img_id, ann_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root (maakt train/ en val/)")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--min_digits", type=int, default=1)
    ap.add_argument("--max_digits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    # digitsize: fraction of image side óf absolute pixels
    ap.add_argument("--size_mode", choices=["fraction", "pixels"], default="pixels",
                    help="Digit size: 'fraction' of image side or absolute 'pixels'")
    ap.add_argument("--min_scale", type=float, default=0.12, help="for size_mode=fraction")
    ap.add_argument("--max_scale", type=float, default=0.22, help="for size_mode=fraction")
    ap.add_argument("--digit_min_px", type=int, default=18, help="for size_mode=pixels")
    ap.add_argument("--digit_max_px", type=int, default=28, help="for size_mode=pixels")

    ap.add_argument("--max_rot", type=float, default=25.0, help="Max rotatie in graden")
    ap.add_argument("--keep_split_jsons", action="store_true",
                    help="Schrijf ook per-split annotations.json (naast het gecombineerde).")
    args = ap.parse_args()

    out_root = Path(args.out)
    train_dir = out_root / "train"
    val_dir   = out_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # MNIST-bron (alleen train-split nodig als bron van digits)
    cache = out_root / "_cache"
    mnist = MNIST(root=str(cache), train=True, download=True)
    mnist_imgs = mnist.data.numpy()      # (N,28,28)
    mnist_lbls = mnist.targets.numpy()   # (N,)

    rng_train = random.Random(args.seed)
    rng_val   = random.Random(args.seed + 1)

    # IDs starten op 1 voor nette COCO
    next_img_id = 1
    next_ann_id = 1

    # Bouw splits
    tr_images, tr_anns, next_img_id, next_ann_id = build_split(
        out_dir=train_dir,
        n_images=args.n_train,
        img_size=args.imgsz,
        mnist_imgs=mnist_imgs,
        mnist_labels=mnist_lbls,
        digits_min=args.min_digits,
        digits_max=args.max_digits,
        rng=rng_train,
        size_mode=args.size_mode,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        digit_min_px=args.digit_min_px,
        digit_max_px=args.digit_max_px,
        max_rot=args.max_rot,
        start_img_id=next_img_id,
        start_ann_id=next_ann_id,
    )
    va_images, va_anns, next_img_id, next_ann_id = build_split(
        out_dir=val_dir,
        n_images=args.n_val,
        img_size=args.imgsz,
        mnist_imgs=mnist_imgs,
        mnist_labels=mnist_lbls,
        digits_min=args.min_digits,
        digits_max=args.max_digits,
        rng=rng_val,
        size_mode=args.size_mode,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        digit_min_px=args.digit_min_px,
        digit_max_px=args.digit_max_px,
        max_rot=args.max_rot,
        start_img_id=next_img_id,
        start_ann_id=next_ann_id,
    )

    # Categorieën: id=1..10, name="0".."9"
    categories = [{"id": i+1, "name": f"{i}"} for i in range(10)]

    # Optioneel: per split JSON (handig voor inspectie)
    if args.keep_split_jsons:
        (train_dir / "annotations.json").write_text(
            json.dumps(make_coco(tr_images, tr_anns, categories), ensure_ascii=False)
        )
        (val_dir / "annotations.json").write_text(
            json.dumps(make_coco(va_images, va_anns, categories), ensure_ascii=False)
        )

    # Gecombineerde JSON op root
    merged_images: List[Dict[str,Any]] = []
    merged_anns  : List[Dict[str,Any]] = []

    # Zet 'train/' en 'val/' prefix in file_name zodat één JSON met beide mappen werkt
    for im in tr_images:
        merged_images.append({
            **im, "file_name": f"train/{im['file_name']}"
        })
    for im in va_images:
        merged_images.append({
            **im, "file_name": f"val/{im['file_name']}"
        })
    merged_anns.extend(tr_anns)
    merged_anns.extend(va_anns)

    combined = make_coco(merged_images, merged_anns, categories)
    (out_root / "annotations.json").write_text(json.dumps(combined, ensure_ascii=False))

    print(f"[DONE] Wrote:")
    print(f"  - {out_root / 'annotations.json'}  (gecombineerd, {len(merged_images)} images, {len(merged_anns)} anns)")
    if args.keep_split_jsons:
        print(f"  - {train_dir / 'annotations.json'}")
        print(f"  - {val_dir / 'annotations.json'}")

if __name__ == "__main__":
    main()
