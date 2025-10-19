"""
Train een Ultralytics YOLOv8m-P6 model op een COCO(-achtig) dataset,
met optioneel een e2cnn-equivariant backbone die inplugt op de YOLO neck/head.

Gebruik:
- Zonder equivariant backbone (zoals voorheen):
  python this_script.py --root /path/to/ds

- Met e2cnn-backbone (rotatiegroep C_n, n=8), en P6 (stride 64) output:
  python this_script.py --root /path/to/ds --use-e2cnn --equiv-n 8 --p6

Vereiste voor equivariantie-optie:
  pip install e2cnn
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
import torch.nn as nn
from loguru import logger
from PIL import Image
from torch.utils.data import Subset as _Subset

# ---- imports eigen code
from train_frcnn_coco_generic import (
    CocoDetGeneric,
    _base_dataset,
    setup_logging,
)

# =========================
# 1) e2cnn BACKBONE (optioneel)
# =========================
# Let op: we importeren e2cnn pas BINNEN de class, zodat het script ook werkt
# als je de flag niet gebruikt of e2cnn (nog) niet geïnstalleerd is.

def _equiv_block(r2, in_type, out_channels, k=3, stride=1):
    """
    Eén 'equivariant' conv-blok met e2cnn:
      - R2Conv met regular representations
      - InnerBatchNorm
      - ReLU

    We kiezen regular reps voor expressiviteit.
    """
    from e2cnn import nn as e2nn
    out_type = e2nn.FieldType(r2, out_channels * [r2.regular_repr])
    pad = nn.ReflectionPad2d(k // 2)              # <— speciaal om rand-FP te voorkomen
    seq = nn.Sequential(
        pad,
        e2nn.R2Conv(in_type, out_type, kernel_size=k, stride=stride, padding=0, bias=False),
        e2nn.InnerBatchNorm(out_type),
        e2nn.ReLU(out_type, inplace=True),
    )
    return seq, out_type



class E2CNNBackbone(nn.Module):
    """
    Equivariant backbone (C_n rotaties) die 3 of 4 feature maps levert:
      - P3 (stride 8), P4 (stride 16), P5 (stride 32), optioneel P6 (stride 64).

    YOLO’s neck/head verwachten gewone PyTorch tensors (N, C, H, W). Daarom:
      1) Intern werken we met e2cnn.GeometricTensor
      2) Aan het eind van elke stage doen we GroupPooling → (trivial reps) → gewone tensor
      3) 1×1 'adapters' om channels te matchen met de YOLO-neck.

    Args:
      n: orde van C_n (bijv. 8)
      in_channels: 3 voor RGB
      base: basisbreedte (64 is prima start)
      out_chs: gewenste out-channels voor (P3, P4, P5, [P6])
      p6: True → lever ook P6-feature (stride 64)
    """

    def __init__(
        self,
        n: int = 8,
        in_channels: int = 3,
        base: int = 64,
        out_chs: Sequence[int] = (256, 512, 1024, 1024),
        p6: bool = False,
    ):
        super().__init__()
        self.p6 = p6
        from e2cnn import gspaces, nn as e2nn

        self.e2nn = e2nn
        self.r2 = gspaces.Rot2dOnR2(N=n)  # C_n (rotaties)

        # Invoer is triviale reps (RGB)
        self.in_type = e2nn.FieldType(self.r2, in_channels * [self.r2.trivial_repr])

        # --- Stem: stride /2
        self.stem, c1 = _equiv_block(self.r2, self.in_type, base, k=3, stride=2)

        # --- Stage 2: naar stride /4
        self.stage2a, c2a = _equiv_block(self.r2, c1, base, k=3, stride=2)
        self.stage2b, c2b = _equiv_block(self.r2, c2a, base, k=3, stride=1)

        # --- Stage 3: naar stride /8 → P3
        self.stage3a, c3a = _equiv_block(self.r2, c2b, base * 2, k=3, stride=2)
        self.stage3b, c3b = _equiv_block(self.r2, c3a, base * 2, k=3, stride=1)

        # --- Stage 4: naar stride /16 → P4
        self.stage4a, c4a = _equiv_block(self.r2, c3b, base * 4, k=3, stride=2)
        self.stage4b, c4b = _equiv_block(self.r2, c4a, base * 4, k=3, stride=1)

        # --- Stage 5: naar stride /32 → P5
        self.stage5a, c5a = _equiv_block(self.r2, c4b, base * 8, k=3, stride=2)
        self.stage5b, c5b = _equiv_block(self.r2, c5a, base * 8, k=3, stride=1)

        # Optioneel P6 (stride /64)
        if self.p6:
            self.stage6a, c6a = _equiv_block(self.r2, c5b, base * 8, k=3, stride=2)
            self.stage6b, c6b = _equiv_block(self.r2, c6a, base * 8, k=3, stride=1)
            self.c6b = c6b

        # GroupPooling (projecteert naar trivial reps → gewone tensor)
        self.pool3 = e2nn.GroupPooling(c3b)
        self.pool4 = e2nn.GroupPooling(c4b)
        self.pool5 = e2nn.GroupPooling(c5b)
        if self.p6:
            self.pool6 = e2nn.GroupPooling(self.c6b)

        # Adapters naar YOLO-neck-in channels (default: 256, 512, 1024[, 1024])
        p3_out, p4_out, p5_out = out_chs[:3]
        self.adapt3 = nn.Conv2d(c3b.size, p3_out, 1)
        self.adapt4 = nn.Conv2d(c4b.size, p4_out, 1)
        self.adapt5 = nn.Conv2d(c5b.size, p5_out, 1)
        if self.p6:
            p6_out = out_chs[3]
            self.adapt6 = nn.Conv2d(self.c6b.size, p6_out, 1)

        # Bewaar types voor forward
        self.c3b = c3b
        self.c4b = c4b
        self.c5b = c5b

    def forward(self, x: torch.Tensor):
        # Wrap naar GeometricTensor
        x = self.e2nn.GeometricTensor(x, self.in_type)

        # Stem + stages
        x = self.stem(x)
        x = self.stage2a(x); x = self.stage2b(x)

        x = self.stage3a(x); x = self.stage3b(x); p3 = self.pool3(x).tensor
        x = self.stage4a(x); x = self.stage4b(x); p4 = self.pool4(x).tensor
        x = self.stage5a(x); x = self.stage5b(x); p5 = self.pool5(x).tensor

        if self.p6:
            x = self.stage6a(x); x = self.stage6b(x); p6 = self.pool6(x).tensor
            return self.adapt3(p3), self.adapt4(p4), self.adapt5(p5), self.adapt6(p6)

        return self.adapt3(p3), self.adapt4(p4), self.adapt5(p5)


def _try_patch_ultralytics_backbone(model, use_p6: bool, equiv_n: int, equiv_base: int):
    """
    Vervang de Ultralytics-backbone door onze e2cnn-backbone.

    - We zoeken het 'backbone'-deel in het Ultralytics graph-model en vervangen dat.
    - De neck/head blijven intact; zolang channel-dimensies kloppen, werkt dit plug-and-play.

    Standaard-neck kanalen (veilig voor m/p6 varianten):
      P3=256, P4=512, P5=1024, P6=1024
    """
    out_chs = (256, 512, 1024, 1024)

    m = model.model  # Ultralytics' DetectionModel (nn.Module)
    target_parent = None
    target_name = None

    # Probeer een child met naam 'backbone'
    for name, child in m.named_children():
        if name.lower().startswith("backbone"):
            target_parent = m
            target_name = name
            break

    # Fallback: vervang index 0 in m.model (ModuleList/Sequential)
    if target_parent is None:
        if hasattr(m, "model") and isinstance(m.model, (nn.ModuleList, nn.Sequential)) and len(m.model) > 0:
            logger.info("Patching Ultralytics graph: replacing model.model[0] with e2cnn backbone")
            e2_backbone = E2CNNBackbone(n=equiv_n, base=equiv_base, out_chs=out_chs, p6=use_p6)
            m.model[0] = e2_backbone
            return True
        else:
            logger.warning("Kon de backbone in Ultralytics model niet vinden; patch wordt overgeslagen.")
            return False

    # Gevonden 'backbone' attribuut: vervang dat
    logger.info(f"Patching Ultralytics graph: replacing '{target_name}' with e2cnn backbone")
    e2_backbone = E2CNNBackbone(n=equiv_n, base=equiv_base, out_chs=out_chs, p6=use_p6)
    setattr(target_parent, target_name, e2_backbone)
    return True


# =========================
# 2) DATA EXPORT + TRAIN (zoals in je script)
# =========================

def _flatten_subset_indices(ds) -> List[int]:
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
        shutil.copy2(src, dst)


def export_coco_to_yolo(
    ds,
    split: str,
    out_dir: Path,
) -> List[int]:
    base_ds: CocoDetGeneric = _base_dataset(ds)
    base_indices = _flatten_subset_indices(ds)
    seen: set[int] = set()
    exported_ids: List[int] = []

    image_root = out_dir / "images" / split
    label_root = out_dir / "labels" / split

    # Bepaal automatisch de offset van cid2idx (0- of 1-based in de bron)
    # We normaliseren altijd naar 0-based voor YOLO.
    min_idx = min(base_ds.cid2idx.values())
    idx_offset = min_idx  # meestal 0 of 1

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

            raw_idx = base_ds.cid2idx.get(ann["category_id"])
            if raw_idx is None:
                continue
            cls = raw_idx - idx_offset  # forceer 0-based
            if cls < 0:
                continue

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        label_path = (label_root / rel_path).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

    return exported_ids



def build_dataset_yaml(out_dir: Path, class_names: Sequence[str]) -> Path:
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
    ap.add_argument("--weights", type=str, default="yolo11n.yaml", help="Modelconfig of YAML (bv. yolo11n.yaml). Gebruik geen .pt bij e2cnn.")
    ap.add_argument("--project", type=str, default="runs", help="Ultralytics projectmap")
    ap.add_argument("--name", type=str, default="y11n_128_full", help="Runnaam binnen projectmap")
    ap.add_argument("--device", type=str, default="", help="Optioneel CUDA device string (bv. '0' of 'cpu')")
    ap.add_argument("--max-steps", type=int, default=0, help="Train maximaal N batches (0 = onbeperkt)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    ap.add_argument("--log-file", default="", help="Pad naar logfile (default: log/train_yolov8m_p6_*.log)")

    # --- NIEUW: e2cnn opties
    ap.add_argument("--use-e2cnn", action="store_true", help="Vervang backbone door e2cnn equivariant backbone")
    ap.add_argument("--equiv-n", type=int, default=8, help="C_n rotatiegroep orde (bijv. 8)")
    ap.add_argument("--equiv-base", type=int, default=64, help="Backbone basisbreedte")
    ap.add_argument("--p6", action="store_true", help="Lever ook P6 (stride 64) feature-map voor -p6 modellen")

    args = ap.parse_args()

    # Logging
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log = log_dir / f"train_yolov8m_p6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, args.log_file or str(default_log))
    logger.info("Logging to {}", args.log_file or default_log)

    logger.info(">>> starting train_yolov8m_p6_coco_generic.py")
    device_str = args.device or ("0" if torch.cuda.is_available() else "cpu")
    logger.info(">>> device request: {}", device_str)

    # Dataset laden
    ann_path = Path(args.ann) if args.ann else None
    train_ds = CocoDetGeneric(args.root, "train", ann_path)
    val_ds = CocoDetGeneric(args.root, "val", ann_path)

    # Optioneel subsetten voor rooktest
    if args.subset < 1.0:
        random.seed(42)

        def make_subset(ds, frac):
            n = max(1, int(len(ds) * frac))
            idx = random.sample(range(len(ds)), n)
            return _Subset(ds, idx)

        train_ds = make_subset(train_ds, args.subset)
        val_ds = make_subset(val_ds, args.subset)
        logger.warning("Subset: {} train + {} val ({}%)", len(train_ds), len(val_ds), int(args.subset * 100))

    base_train = _base_dataset(train_ds)

    # Tijdelijke YOLO-structuur opbouwen
    with tempfile.TemporaryDirectory(prefix="yolo_coco_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        export_coco_to_yolo(train_ds, "train", tmpdir_path)
        export_coco_to_yolo(val_ds, "val", tmpdir_path)

        class_names = [name for _, name in sorted(base_train.idx2name.items())]
        data_yaml = build_dataset_yaml(tmpdir_path, class_names)
        logger.info("Dataset YAML: {}", data_yaml)

        from ultralytics import YOLO

        # 1) Bouw model uit YAML (geen .pt)
        #    Zorg dat --weights een YAML is, bv. yolo11n.yaml
        model = YOLO(args.weights)

        # Forceer het bouwen van de onderliggende nn.Module graph
        _ = model.model  # triggert constructie

        # 2) Patch hier je e2cnn-backbone IN-PLACE (indien aangevraagd)
        if args.use_e2cnn:
            logger.info(">> e2cnn backbone patch aangevraagd: C_n n={}, base={}, p6={}",
                        args.equiv_n, args.equiv_base, args.p6)
            try:
                patched_ok = _try_patch_ultralytics_backbone(
                    model=model,              # let op: geef de YOLO wrapper; de functie pakt intern model.model
                    use_p6=args.p6,
                    equiv_n=args.equiv_n,
                    equiv_base=args.equiv_base,
                )
                if not patched_ok:
                    logger.warning("e2cnn patch niet toegepast; ga door met standaard backbone.")
            except Exception as e:
                logger.exception("e2cnn patch faalde met uitzondering: {}", e)
                logger.warning("We gaan door met de standaard backbone.")

        # # 3) Wrap de (mogelijk gepatchte) graph terug in YOLO zónder checkpoint
        # patched_nn = model.model          # nn.Module (DetectionModel met gepatchte backbone)
        # # model = YOLO(patched_nn)
        # model = YOLO(model=patched_nn, task="detect")  # <— belangrijk: keywords!
        # model.ckpt = {"model": patched_nn}
        try:
            # Loguru gebruikt {} i.p.v. %s
            logger.info("Backbone[0] type at train(): {}", model.model.model[0].__class__.__name__)
        except Exception as e:
            logger.warning("Kon backbone type niet loggen: {}", e)

        # (optioneel) extra check
        from inspect import isclass
        if model.model and hasattr(model.model, "model"):
            is_e2 = model.model.model and model.model.model[0].__class__.__name__ == "E2CNNBackbone"
            logger.info("Is E2CNNBackbone actief? {}", is_e2)

        # 4) Train-kwargs en augmentaties (optioneel uitzetten)
        train_kwargs = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,

            "optimizer": "SGD",
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 5.0,
            "cos_lr": True,


            "imgsz": args.imgsz,
            "project": args.project,
            "name": args.name,
            "device": args.device or device_str,
            "exist_ok": True,
            # heel belangrijk bij gepatchte backbone:
            "pretrained": False,   # << voorkomt proberen .pt-weights te laden (die sleutels mismatchen)
            "resume": False,

            # Als je GEEN YOLO augmentaties wilt:
            "mosaic": 0, "mixup": 0, "copy_paste": 0,
            "degrees": 0, "translate": 0, "scale": 0, "shear": 0, "perspective": 0,
            "flipud": 0, "fliplr": 0,
            "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
            "erasing": 0,

            # --- matcher & losses ---
            "iou": 0.5,        # soepeler label assignment
            "fl_gamma": 1.5,   # minder FP's
            "box": 7.5,        # iets lager
            "cls": 0.8,        # terug naar normaal gewicht
            "label_smoothing": 0.05,  # optioneel, kan ook op 0.0
            
        }
        if args.max_steps > 0:
            train_kwargs["max_train_batches"] = args.max_steps

        logger.info("Ultralytics train kwargs: {}", json.dumps(train_kwargs, indent=2))

        # Train
        results = model.train(**train_kwargs)
        logger.info("Train results: {}", results)

        # Val
        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device or device_str, conf=0.25, iou=0.6)
        logger.info("Validation metrics: {}", metrics)

    logger.success("Done")


if __name__ == "__main__":
    main()



# VB. Aanroepen:
#  python .\train_equiv_yolo.py `
#   --root C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\processed\mnist_det_128 `
#   --ann  C:\Users\Falco\Documents\Python_project\MADS_thesis_equivariance\data\processed\mnist_det_128\annotations.json `
#   --weights yolo11n.yaml `
#   --imgsz 256 `
#   --epochs 100 `
#   --batch 16 `
#   --name y11n_256_e2cnn_adamw `
#   --device 0 `
#   --use-e2cnn `
#   --equiv-n 8 `
#   --equiv-base 64 `

