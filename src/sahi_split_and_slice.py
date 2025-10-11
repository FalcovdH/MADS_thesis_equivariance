
import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple

# SAHI imports
try:
    from sahi.slicing import slice_coco
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'sahi'. Install with:\n\n    pip install sahi pillow\n"
    ) from e

def load_coco(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in ["images", "annotations", "categories"]:
        if k not in data:
            raise ValueError(f"COCO JSON missing '{k}'")
    return data

def write_coco(path: Path, images: List[Dict], annotations: List[Dict], categories: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f, ensure_ascii=False, indent=2)

def deterministic_split(images: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    imgs = images.copy()
    rng.shuffle(imgs)
    n_val = int(round(len(imgs) * val_ratio))
    val = imgs[:n_val]
    train = imgs[n_val:]
    return train, val

def subset_coco(coco: Dict, subset_images: List[Dict]) -> Dict:
    subset_ids = {img["id"] for img in subset_images}
    anns = [a for a in coco["annotations"] if a["image_id"] in subset_ids]
    # Reindex image ids sequentially if desired (not required by SAHI). We'll keep original ids to preserve linkage.
    return {"images": subset_images, "annotations": anns, "categories": coco["categories"]}

def run_split_and_slice(
    images_dir: Path,
    coco_json: Path,
    out_dir: Path,
    val_ratio: float,
    slice_size: int,
    overlap: float,
    seed: int,
    ignore_negative_samples: bool,
):
    coco = load_coco(coco_json)

    # 1) Split by image BEFORE slicing
    train_imgs, val_imgs = deterministic_split(coco["images"], val_ratio, seed)

    # 2) Build temporary COCO JSONs for each split
    tmp_dir = out_dir / "_tmp_splits"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_coco_path = tmp_dir / "train.json"
    val_coco_path = tmp_dir / "val.json"

    train_coco = subset_coco(coco, train_imgs)
    val_coco = subset_coco(coco, val_imgs)

    write_coco(train_coco_path, train_coco["images"], train_coco["annotations"], train_coco["categories"])
    write_coco(val_coco_path, val_coco["images"], val_coco["annotations"], val_coco["categories"])

    # 3) Slice each split with SAHI
    #    SAHI will create images under output_dir/<images_dir_name> by default; we'll set per-split directories.
    train_out = out_dir / "train"
    val_out = out_dir / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    # SAHI uses slice_height/width and overlap ratios per dimension
    # We propagate the user's settings to both height and width.
    # We also set output_coco_annotation_file_name to the standard 'annotations.json'.

    for split_name, split_json, split_out in [
        ("train", str(train_coco_path), train_out),
        ("val", str(val_coco_path), val_out),
    ]:
        print(f"[INFO] Slicing {split_name} with SAHI -> {split_out}")
        slice_coco(
            coco_annotation_file_path=split_json,
            image_dir=str(images_dir),
            output_coco_annotation_file_name="annotations.json",
            output_dir=str(split_out),
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            ignore_negative_samples=ignore_negative_samples,
            verbose=True,
        )

    print("\n[DONE] Wrote sliced dataset to:")
    print(f"  {train_out}  (train)")
    print(f"  {val_out}    (val)")
    print("\nEach split contains:")
    print("  - images/            (sliced tiles)")
    print("  - annotations.json   (COCO annotations for sliced tiles)")

def main():
    # === handmatige instellingen ===
    images_dir = Path(r"C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\raw\images")  # map met originele afbeeldingen
    coco_json = Path(r"C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\raw\annotations_coco.json")  # COCO-bestand van de converter
    out_dir = Path(r"C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\processed\images")  # outputmap

    val_ratio = 0.2        # 20% validatie, 80% training
    slice_size = 1280      # slice-afmeting in pixels
    overlap = 0.2          # overlap tussen tiles
    seed = 42              # voor deterministische splitsing
    ignore_negative_samples = True  # oversla tiles zonder annotaties

    # === run de slicer ===
    run_split_and_slice(
        images_dir=images_dir,
        coco_json=coco_json,
        out_dir=out_dir,
        val_ratio=val_ratio,
        slice_size=slice_size,
        overlap=overlap,
        seed=seed,
        ignore_negative_samples=ignore_negative_samples,
    )

if __name__ == "__main__":
    main()

