# src/train_frcnn_resnet50.py
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image

TARGET_NAMES = ["chalk","condensation","discoloration","liquid","product"]

def find_annotations(split_dir: Path) -> Path:
    for n in ["annotations.json","annotations.coco.json","annotations.json_coco.json","instances_default.json","combined_coco.json"]:
        p = split_dir / n
        if p.exists():
            return p
    for p in split_dir.glob("*.json"):
        return p
    raise FileNotFoundError(f"No COCO json in {split_dir}")

class CocoDet5(Dataset):
    def __init__(self, root: str, split: str):
        self.split = split
        self.split_dir = Path(root) / split
        ann_path = find_annotations(self.split_dir)
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        # map categorie-id -> index (1..5)
        name2cid = {c["name"]: c["id"] for c in data["categories"]}
        self.cid2idx = {name2cid[n]: i+1 for i, n in enumerate(TARGET_NAMES) if n in name2cid}  # 1..5

        self.id2img = {im["id"]: im for im in data["images"]}
        self.anns_by_img = {}
        for a in data.get("annotations", []):
            if a["category_id"] in self.cid2idx:
                self.anns_by_img.setdefault(a["image_id"], []).append(a)

        self.img_dir_pref = self.split_dir / "images"
        self.to_tensor = transforms.ToTensor()

        print(f"[{split}] images={len(self.id2img)}  classes={list(self.cid2idx.values())}")

    def __len__(self): 
        return len(self.id2img)

    def _img_path(self, fname: str) -> Path:
        p1 = self.img_dir_pref / fname
        if p1.exists():
            return p1
        p2 = self.split_dir / fname
        if p2.exists():
            return p2
        return p1

    def __getitem__(self, idx: int):
        img_id = list(self.id2img.keys())[idx]
        info = self.id2img[img_id]
        img = Image.open(self._img_path(info["file_name"])).convert("RGB")

        boxes, labels = [], []
        for a in self.anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x+w, y+h])
            labels.append(self.cid2idx[a["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return self.to_tensor(img), target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

if __name__ == "__main__":
    print(">>> starting train_frcnn_resnet50.py")  # extra zichtbare start-print
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device: {device}")

    train_ds = CocoDet5(args.root, "train")
    val_ds   = CocoDet5(args.root, "val")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    num_classes = 5 + 1
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", box_score_thresh=0.05)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    model.to(device)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    best = 1e9
    for e in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, optimizer, device)
        dt = time.time() - t0
        print(f"[{e+1:02d}/{args.epochs}] loss={loss:.4f}  time={dt:.1f}s")
        Path("runs_frcnn_r50").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "runs_frcnn_r50/last.pth")
        if loss < best:
            best = loss
            torch.save(model.state_dict(), "runs_frcnn_r50/best.pth")
    print("✅ Done -> runs_frcnn_r50/best.pth")
