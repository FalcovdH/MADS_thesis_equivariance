# src/train_frcnn_coco_generic.py
import argparse, json, time, random, tempfile
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image

# COCO eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils.data import Subset as _Subset

# ---------------- Loguru ----------------
from loguru import logger
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configureer Loguru console + (optioneel) file logging."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        enqueue=True,
    )
    if log_file:
        logger.add(
            log_file,
            level=level.upper(),
            rotation="50 MB",
            retention="10 days",
            compression="zip",
            enqueue=True,
        )


def _base_dataset(ds):
    """Pak de onderliggende dataset (ook als ds een torch.utils.data.Subset is)."""
    return ds.dataset if isinstance(ds, _Subset) else ds


def find_annotations(split_dir: Path) -> Path:
    for n in ["annotations.json", "annotations.coco.json", "annotations.json_coco.json", "instances_default.json", "combined_coco.json"]:
        p = split_dir / n
        if p.exists():
            return p
    for p in split_dir.glob("*.json"):
        return p
    raise FileNotFoundError(f"No COCO json in {split_dir}")


class CocoDetGeneric(Dataset):
    """
    Werkt met:
      - gecombineerde JSON in root (file_name 'train/...', 'val/...')
      - per-split JSON in train/ of val/
    """
    def __init__(self, root: str, split: str, ann_path: Path | None = None):
        self.root = Path(root)
        self.split = split

        if ann_path is None:
            default_split_json = self.root / split / "annotations.json"
            ann_path = default_split_json if default_split_json.exists() else self.root / "annotations.json"
        self.ann_path = Path(ann_path)
        data = json.loads(self.ann_path.read_text(encoding="utf-8"))

        cats = data.get("categories", [])
        self.cid_sorted = [c["id"] for c in cats]
        self.cid2idx = {cid: i + 1 for i, cid in enumerate(self.cid_sorted)}  # original COCO id -> 1..K
        self.idx2cid = {i + 1: cid for i, cid in enumerate(self.cid_sorted)}  # 1..K -> original COCO id
        self.idx2name = {i + 1: c["name"] for i, c in enumerate(cats)}
        self.num_classes = len(self.cid2idx) + 1  # + background

        self.id2img = {im["id"]: im for im in data["images"]}
        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for a in data.get("annotations", []):
            if a["category_id"] in self.cid2idx:
                self.anns_by_img.setdefault(a["image_id"], []).append(a)

        # is combined? (file_name met prefix 'train/' of 'val/')
        self.ann_is_combined = any(('/' in im["file_name"] or '\\' in im["file_name"]) for im in self.id2img.values())

        self.to_tensor = transforms.ToTensor()

        # filter op split als combined-json
        if self.ann_is_combined and split in ("train", "val"):
            pref = f"{split}/"
            keep = {iid for iid, im in self.id2img.items() if im["file_name"].replace("\\", "/").startswith(pref)}
            self.id2img = {iid: self.id2img[iid] for iid in keep}
            self.anns_by_img = {iid: self.anns_by_img.get(iid, []) for iid in self.id2img.keys()}

        logger.info("[{}] images={}  nc={}  names={}", split, len(self.id2img), self.num_classes - 1,
                    [self.idx2name[i] for i in sorted(self.idx2name)])

    def __len__(self):
        return len(self.id2img)

    def _resolve_img_path(self, file_name: str) -> Path:
        fn = file_name.replace("\\", "/")
        # 1) Combined JSON: probeer direct en daarna met 'images/'
        if self.ann_is_combined:
            p = self.root / fn
            if p.exists():
                return p
            # fn = "train/xxx.png" -> probeer "root/train/images/xxx.png"
            parts = fn.split("/", 1)
            if len(parts) == 2:
                split_name, rel = parts
                p2 = self.root / split_name / "images" / rel
                if p2.exists():
                    return p2
        # 2) Split JSON: probeer split/images/ en split/
        p1 = self.root / self.split / "images" / file_name
        if p1.exists():
            return p1
        p2 = self.root / self.split / file_name
        if p2.exists():
            return p2
        # 3) Fallback: root/images/
        p3 = self.root / "images" / file_name
        if p3.exists():
            return p3
        raise FileNotFoundError(f"Image not found: {file_name}")

    def __getitem__(self, idx: int):
        img_id = list(self.id2img.keys())[idx]
        info = self.id2img[img_id]
        img = Image.open(self._resolve_img_path(info["file_name"])).convert("RGB")

        boxes, labels = [], []
        for a in self.anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cid2idx[a["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return self.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total = 0.0
    for step, (imgs, tgts) in enumerate(loader, 1):
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        loss = sum(model(imgs, tgts).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        if max_steps and step >= max_steps:
            break
    seen = min(len(loader), max_steps or len(loader))
    return total / max(1, seen)


@torch.no_grad()
def evaluate_coco(model, ds: "CocoDetGeneric", dl: DataLoader, device, ann_path: Path):
    """COCO mAP evaluatie: retourneer dict met mAP, AP50, AP75, AR.
       Patcht GT-json met lege 'info' en 'licenses' als die ontbreken."""
    model.eval()
    results = []

    for imgs, tgts in dl:
        imgs = [i.to(device) for i in imgs]
        outputs = model(imgs)
        for out, tgt in zip(outputs, tgts):
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].cpu().tolist()
            scores = out["scores"].cpu().tolist()
            labels = out["labels"].cpu().tolist()
            for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels):
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                results.append({
                    "image_id": img_id,
                    # terug naar originele COCO cat id:
                    "category_id": ds.idx2cid.get(int(lab), 1),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(sc),
                })

    # ---- GT JSON patchen met lege info/licenses indien nodig ----
    gt = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    if "info" not in gt:
        gt["info"] = {"description": "generated"}
    if "licenses" not in gt:
        gt["licenses"] = []
    # veiligheid: ids als int
    for im in gt.get("images", []):
        if isinstance(im.get("id", None), str):
            im["id"] = int(im["id"])
    for an in gt.get("annotations", []):
        if isinstance(an.get("id", None), str):
            an["id"] = int(an["id"])
        if isinstance(an.get("image_id", None), str):
            an["image_id"] = int(an["image_id"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fgt:
        json.dump(gt, fgt)
        gt_path = fgt.name

    coco_gt = COCO(gt_path)

    if len(results) == 0:
        logger.warning("Geen detections; returning zeros for COCO metrics")
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fres:
        json.dump(results, fres)
        res_path = fres.name

    coco_dt = coco_gt.loadRes(res_path)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "mAP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "AR": float(ev.stats[8]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root met train/, val/ en evt. root/annotations.json")
    ap.add_argument("--ann", type=str, default="", help="Optioneel pad naar JSON (bv. root/annotations.json). Leeg = auto-detect.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)       # CPU-vriendelijk
    ap.add_argument("--workers", type=int, default=0)     # Windows CPU: 0
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--subset", type=float, default=1.0, help="0..1 fractie van data per split")
    ap.add_argument("--max_steps", type=int, default=0, help="0=geen limiet; anders max batches per epoch")
    ap.add_argument("--fast", action="store_true", help="Speed knobs (minder proposals / kleinere batches / backbone freeze)")
    # Logging
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    ap.add_argument("--log-file", default="", help="Pad naar logfile (optioneel, bv. runs/train.log)")

    args = ap.parse_args()

    # maak standaard logmap + bestandsnaam
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log = str(log_dir / f"train_frcnn_coco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    log_file = args.log_file or default_log
    setup_logging(args.log_level, log_file)
    logger.info("Logging to {}", log_file)

    logger.info(">>> starting train_frcnn_coco_generic.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(">>> device: {}", device)

    ann_path = Path(args.ann) if args.ann else None
    train_ds = CocoDetGeneric(args.root, "train", ann_path)
    val_ds = CocoDetGeneric(args.root, "val", ann_path)

    # subset (sneller testen)
    if args.subset < 1.0:
        random.seed(42)

        def sub(ds, frac):
            n = max(1, int(len(ds) * frac))
            idx = random.sample(range(len(ds)), n)
            return Subset(ds, idx)

        train_ds = sub(train_ds, args.subset)
        val_ds = sub(val_ds, args.subset)
        logger.warning("Subset: {} train + {} val ({}%)", len(train_ds), len(val_ds), int(args.subset * 100))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    # onderliggende datasets (voor attrs zoals num_classes / ann_path / idx2cid)
    _base_train = _base_dataset(train_ds)
    _base_val = _base_dataset(val_ds)

    # model
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", box_score_thresh=0.05)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = _base_train.num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    # FAST-modus (rooktest)
    if args.fast:
        if hasattr(model.rpn, "batch_size_per_image"):
            model.rpn.batch_size_per_image = 64
        if hasattr(model.roi_heads, "batch_size_per_image"):
            model.roi_heads.batch_size_per_image = 128
        for attr, val in [("pre_nms_top_n_train", 1000), ("post_nms_top_n_train", 200),
                          ("pre_nms_top_n_test", 600), ("post_nms_top_n_test", 100)]:
            if hasattr(model.rpn, attr):
                setattr(model.rpn, attr, val)
        if hasattr(model.roi_heads, "detections_per_img"):
            model.roi_heads.detections_per_img = 50
        try:
            for p in model.backbone.body.parameters():
                p.requires_grad = False
            logger.info(">>> FAST: backbone frozen; only heads trainen")
        except Exception:
            pass

    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    best = 1e9
    for e in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, optimizer, device, max_steps=(args.max_steps or None))
        logger.info("[{}/{}] train_loss={:.4f}  time={:.1f}s", e + 1, args.epochs, loss, time.time() - t0)

        # === COCO eval op val-set ===
        ann_for_eval = _base_val.ann_path
        metrics = evaluate_coco(model, _base_val, val_dl, device, ann_for_eval)
        logger.info("[val] mAP={:.3f}  AP50={:.3f}  AP75={:.3f}  AR={:.3f}",
                    metrics["mAP"], metrics["AP50"], metrics["AP75"], metrics["AR"])

    logger.success("Done")
