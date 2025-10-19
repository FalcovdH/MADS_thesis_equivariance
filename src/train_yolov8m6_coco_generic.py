"""Train een Ultralytics YOLOv8m-P6 model op een COCO(-achtig) dataset.

De script sluit qua CLI grotendeels aan bij ``train_frcnn_coco_generic.py``
zodat beide pipelines onderling vergelijkbaar blijven. De belangrijkste
stappen zijn:

1. Laden van dezelfde ``CocoDetGeneric`` dataset klasse.
2. (Optioneel) sub-samplen voor snelle rooktesten.
3. Exporteren naar het YOLO-formaat (``images/`` + ``labels/``) in een
   tijdelijke werkomgeving met symbolische links.
4. Starten van een standaard ``YOLO.train`` run met de
   ``yolov8m-p6.pt`` foundation weights.

Het script logt naar zowel stdout als optioneel een logfile met Loguru en
rapporteert na afloop de belangrijkste metriek uit ``model.train`` en
``model.val``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Subset as _Subset

from train_frcnn_coco_generic import (
    CocoDetGeneric,
    _base_dataset,
    setup_logging,
)


def _flatten_subset_indices(ds) -> List[int]:
    """Zet (geneste) ``torch.utils.data.Subset`` om naar indices t.o.v. de
    onderliggende ``CocoDetGeneric`` dataset.
    """

    if isinstance(ds, _Subset):
        parent_indices = _flatten_subset_indices(ds.dataset)
        return [parent_indices[i] for i in ds.indices]
    return list(range(len(ds)))


def _ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
       return
    try:
        dst.symlink_to(src)
    except OSError:
        # Fallback voor filesystems die geen symlinks ondersteunen.
        shutil.copy2(src, dst)


def export_coco_to_yolo(
    ds,
    split: str,
    out_dir: Path,
    class_offset: int = 1,
) -> List[int]:
    """Exporteer een (subset van een) ``CocoDetGeneric`` dataset naar het
    YOLO-detectieformaat.

    Parameters
    ----------
    ds:
        ``CocoDetGeneric`` of ``Subset`` daaruit.
    split:
        ``"train"`` of ``"val"`` – bepaalt de submap onder ``images/`` en
        ``labels/``.
    out_dir:
        Basismap waar ``images/`` en ``labels/`` aangemaakt worden.
    class_offset:
        ``CocoDetGeneric`` labelt classes 1..K; voor YOLO hebben we een
        0..K-1 indeling nodig. ``class_offset`` geeft het verschil aan.

    Returns
    -------
    List[int]
       De unieke COCO ``image_id`` waarden die geëxporteerd zijn.
    """

    base_ds: CocoDetGeneric = _base_dataset(ds)
    base_indices = _flatten_subset_indices(ds)
    seen: set[int] = set()
    exported_ids: List[int] = []

    image_root = out_dir / "images" / split
    label_root = out_dir / "labels" / split

    for base_idx in base_indices:
        ids_list = getattr(base_ds, "ids", list(base_ds.id2img.keys()))
        img_id = ids_list[base_idx]
        if img_id in seen:
            continue
        seen.add(img_id)
        exported_ids.append(img_id)

        img_info = base_ds.id2img[img_id]
        file_name = img_info["file_name"].replace("\\", "/")
        rel_path = Path(file_name)

        src_img = base_ds._resolve_img_path(img_info["file_name"])
        dst_img = image_root / rel_path
        _ensure_symlink(src_img, dst_img)

        width = img_info.get("width")
        height = img_info.get("height")
        if not width or not height:
            with Image.open(src_img) as im:
                width, height = im.size

        lines: List[str] = []
        for ann in base_ds.anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            ww = w / width
            hh = h / height

            cls = base_ds.cid2idx.get(ann["category_id"], 0) - class_offset
            if cls < 0:
                continue

            lines.append(
                f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"
            )

        label_path = (label_root / rel_path).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

    return exported_ids


def build_dataset_yaml(out_dir: Path, class_names: Sequence[str]) -> Path:
    """Genereer een minimalistische ``dataset.yaml`` voor Ultralytics."""

    yaml_lines = [f"path: {out_dir}", "train: images/train", "val: images/val", "names:"]
    for idx, name in enumerate(class_names):
        yaml_lines.append(f"  {idx}: {name}")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root met train/ en val/ mappen")
    ap.add_argument("--ann", type=str, default="", help="Optioneel pad naar gecombineerde annotations.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initiële learning rate voor Ultralytics trainer")
    ap.add_argument("--imgsz", type=int, default=128, help="Invoerresolutie")
    ap.add_argument("--subset", type=float, default=1.0, help="0..1 fractie van data per split")
    ap.add_argument("--weights", type=str, default="yolov8m-p6.pt", help="Start-weights (lokaal pad of modelnaam)")
    ap.add_argument("--project", type=str, default="runs", help="Ultralytics projectmap")
    ap.add_argument("--name", type=str, default="train_yolov8m_p6", help="Runnaam binnen projectmap")
    ap.add_argument("--device", type=str, default="", help="Optioneel CUDA device string (bv. '0' of 'cpu')")
    ap.add_argument("--max-steps", type=int, default=0, help="Train maximaal N batches (0 = onbeperkt)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    ap.add_argument("--log-file", default="", help="Pad naar logfile (default: log/train_yolov8m_p6_*.log)")

    args = ap.parse_args()

    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log = log_dir / f"train_yolov8m_p6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, args.log_file or str(default_log))
    logger.info("Logging to {}", args.log_file or default_log)

    logger.info(">>> starting train_yolov8m_p6_coco_generic.py")
    device_str = args.device or ("0" if torch.cuda.is_available() else "cpu")
    logger.info(">>> device request: {}", device_str)

    ann_path = Path(args.ann) if args.ann else None
    train_ds = CocoDetGeneric(args.root, "train", ann_path)
    val_ds = CocoDetGeneric(args.root, "val", ann_path)

    if args.subset < 1.0:
        random.seed(42)

        def make_subset(ds, frac):
            n = max(1, int(len(ds) * frac))
            idx = random.sample(range(len(ds)), n)
            return _Subset(ds, idx)

        train_ds = make_subset(train_ds, args.subset)
        val_ds = make_subset(val_ds, args.subset)
        logger.warning(
            "Subset: {} train + {} val ({}%)",
            len(train_ds),
            len(val_ds),
            int(args.subset * 100),
        )

    base_train = _base_dataset(train_ds)

    with tempfile.TemporaryDirectory(prefix="yolo_coco_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        export_coco_to_yolo(train_ds, "train", tmpdir_path)
        export_coco_to_yolo(val_ds, "val", tmpdir_path)

        class_names = [name for _, name in sorted(base_train.idx2name.items())]
        data_yaml = build_dataset_yaml(tmpdir_path, class_names)
        logger.info("Dataset YAML: {}", data_yaml)

        from ultralytics import YOLO

        model = YOLO(args.weights)

        train_kwargs = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "lr0": args.lr0,
            "imgsz": args.imgsz,
            "project": args.project,
            "name": args.name,
            "device": args.device or device_str,
            "exist_ok": True,
        }

        if args.max_steps > 0:
            train_kwargs["max_train_batches"] = args.max_steps

        logger.info("Ultralytics train kwargs: {}", json.dumps(train_kwargs, indent=2))
        results = model.train(**train_kwargs)
        logger.info("Train results: {}", results)

        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device or device_str)
        logger.info("Validation metrics: {}", metrics)

    logger.success("Done")


if __name__ == "__main__":
    main()


# Runnen script:

# python .\train_yolov8m6_coco_generic.py `
#    --root C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\processed\mnist_det_128 `
#    --ann  C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\processed\mnist_det_128\annotations.json `
#    --weights yolo11n.pt `  
#    --imgsz 128 `
#    --epochs 100 `
#    --batch 16 `
#    --name y11n_128_full_251019_14.26 `
#    --device 0