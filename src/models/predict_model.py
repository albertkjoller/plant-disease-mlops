import torch
from src.models.model import ImageClassification
from src.data.dataloader import PlantVillage
import time
from omegaconf import OmegaConf
import torch
from pytorch_lightning import Trainer
import hydra
from hydra.utils import to_absolute_path
import argparse

#############################
# fix for path, but very ugly
from pathlib import Path
import os
import sys

myDir = os.getcwd()
path = Path(f"{myDir}/app")
a = str(path.parent.absolute())
sys.path.append(a)
##############################
# from deployment.app.app_utils import get_base_model
import os

import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.1")
def predict(config) -> None:
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument(
        "--model_checkpoint", default="epoch=00-val_acc=0.69-13-01-2023 22:45:11.ckpt"
    )
    args = parser.parse_args()
    print(args)
    print(f"\nRunning on CUDA? {torch.cuda.is_available()}")
    print(f"\nConfiguration: \n {OmegaConf.to_yaml(config)}")

    device, accelerator_type, num_devices = (
        (torch.device("cuda"), "gpu", -1)
        if torch.cuda.is_available()
        else (torch.device("cpu"), "cpu", None)
    )

    # Extract information from configuration
    experiment = config.experiment
    paths = config.paths
    loggers = config.logging

    testData = PlantVillage(
        dtype="test",
        data_path=to_absolute_path(paths.data_path),
        process_type=experiment.data.process_type,
    )
    test_loader = testData.get_loader(
        batch_size=experiment.training.batch_size,
        shuffle=False,
        num_workers=experiment.data.num_workers,
    )

    # Define trainer
    trainer = Trainer(
        max_epochs=experiment.training.epochs,
        accelerator=accelerator_type,
        devices=num_devices,
    )

    # Initialize model
    model = ImageClassification.load_from_checkpoint(args.model_checkpoint)
    model = model.to(device)

    predictions = trainer.predict(model, test_loader)
    predictions = torch.cat([x for x in predictions])
    labels = torch.cat([x["label"] for x in test_loader])
    test_acc = torch.mean((predictions == labels).float())
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    predict()