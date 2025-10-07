import torch, time, argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import json

# --- Dataset ---
class CocoDet5(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / split
        ann = self.root / "annotations.json"
        with open(ann) as f:
            data = json.load(f)
        name2id = {c["name"]: c["id"] for c in data["categories"]}
        self.target_names = ["chalk","condensation","discoloration","liquid","product"]
        self.valid_ids = {name2id[n] for n in self.target_names if n in name2id}
        self.id2img = {im["id"]: im for im in data["images"]}
        self.anns = {}
        for a in data["annotations"]:
            if a["category_id"] in self.valid_ids:
                self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self): return len(self.id2img)

    def __getitem__(self, idx):
        img_id = list(self.id2img.keys())[idx]
        info = self.id2img[img_id]
        img = Image.open(Path(self.root, "images", info["file_name"])).convert("RGB")
        boxes, labels = [], []
        for a in self.anns.get(img_id, []):
            x,y,w,h = a["bbox"]
            boxes.append([x,y,x+w,y+h])
            labels.append(1)  # all single-class for demo; fix later if needed
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        img = transforms.ToTensor()(img)
        return img, target

def collate_fn(batch): return tuple(zip(*batch))

# --- Training ---
def train_one_epoch(model, loader, opt, device):
    model.train(); total=0
    for imgs, tgts in loader:
        imgs=[i.to(device) for i in imgs]
        tgts=[{k:v.to(device) for k,v in t.items()} for t in tgts]
        loss_dict=model(imgs,tgts)
        loss=sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        total+=loss.item()
    return total/len(loader)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--epochs",type=int,default=10)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds=CocoDet5(args.root,"train")
    val_ds=CocoDet5(args.root,"val")
    train_dl=DataLoader(train_ds,batch_size=2,shuffle=True,collate_fn=collate_fn)
    val_dl=DataLoader(val_ds,batch_size=2,shuffle=False,collate_fn=collate_fn)

    model=fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_feats=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_feats,6) # 5 classes + background
    model.to(device)
    opt=torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.9,weight_decay=1e-4)

    for e in range(args.epochs):
        loss=train_one_epoch(model,train_dl,opt,device)
        print(f"Epoch {e+1}: loss={loss:.3f}")
    torch.save(model.state_dict(),"frcnn_resnet50_baseline.pth")
    print("✅ Done")
