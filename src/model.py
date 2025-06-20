import torch
from torch import nn
from e2cnn import gspaces
from e2cnn import nn as e2nn


class SE2EquivariantModel(nn.Module):
    def __init__(self, n_rotations: int = 8, hidden: int = 64, n_classes: int = 2):
        super().__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=n_rotations)

        # Input: 1 channel grayscale image
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        out_type = e2nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
            e2nn.ReLU(out_type, inplace=True),
            e2nn.PointwiseMaxPool(out_type, kernel_size=2)
        )

        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(out_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.ReLU(out_type, inplace=True),
            e2nn.PointwiseMaxPool(out_type, kernel_size=2)
        )

        self.gpool = e2nn.GroupPooling(out_type)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * (64 // 4) * (64 // 4), n_classes)

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.block1.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gpool(x)
        x = self.flatten(x.tensor)
        x = self.fc(x)
        return x


def make_model(config: dict = None) -> nn.Module:
    config = config or {}
    return SE2EquivariantModel(
        n_rotations=config.get("n_rotations", 8),
        hidden=config.get("hidden", 64),
        n_classes=config.get("n_classes", 2)
    )