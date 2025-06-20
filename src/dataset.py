from pathlib import Path
from typing import Literal
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

LABELS = {
    "discoloration": 0,
    "liquid": 1
}

class PatchDataset(Dataset):
    def __init__(
        self,
        path: Path,
        image_size: int = 64,
        grayscale: bool = True,
    ) -> None:
        self.paths = list(Path(path).glob("*.png"))
        self.image_size = image_size
        self.grayscale = grayscale

        self.transform = self._get_transform()

    def _get_transform(self):
        base = [transforms.Resize((self.image_size, self.image_size))]
        if self.grayscale:
            base.append(transforms.Grayscale(num_output_channels=1))
        base.append(transforms.ToTensor())
        return transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label_name = self._extract_label_from_filename(img_path.name)
        label = torch.tensor(LABELS[label_name], dtype=torch.long)

        image = Image.open(img_path)
        tensor = self.transform(image)

        return tensor, label

    def _extract_label_from_filename(self, filename: str) -> str:
        for label in LABELS:
            if label in filename:
                return label
        raise ValueError(f"Onbekend label in bestandsnaam: {filename}")

    def __repr__(self) -> str:
        return f"PatchDataset(size={len(self)}, labels={list(LABELS.keys())})"
