import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from loguru import logger

from src.dataset import PatchDataset
from src.model import make_model
from src.metric import Accuracy  # zelfde opzet als jouw voorbeeld


def train_model(config):
    data_path = Path(config["data_dir"])
    dataset = PatchDataset(path=data_path, image_size=64)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = make_model(config)  # zelf aan te maken
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # evaluatie
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        tune.report(accuracy=accuracy)


if __name__ == "__main__":
    data_dir = "data/patches"

    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden": tune.randint(32, 128),
        "data_dir": data_dir
    }

    reporter = CLIReporter()
    reporter.add_metric_column("accuracy")

    scheduler = ASHAScheduler(metric="accuracy", mode="max")

    analysis = tune.run(
        train_model,
        config=config,
        resources_per_trial={"cpu": 2, "gpu": 0},
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="ray_results",
    )
