import os
from pathlib import Path

import time
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloader import PlantVillage
from src.models.model import ImageClassification

import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.1")
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Extract information from configuration
    experiment = config.experiment
    paths = config.paths
    loggers = config.logging

    # Define compute scenario
    device, accelerator_type, num_devices = (
        (torch.device("cuda"), "gpu", -1)
        if torch.cuda.is_available()
        else (torch.device("cpu"), "cpu", 0)
    )

    # Create torch DataLoader for training set
    trainData = PlantVillage(
        dtype="train",
        data_path=to_absolute_path(paths.data_path),
        process_type=experiment.data.process_type,
    )
    train_loader = trainData.get_loader(
        batch_size=experiment.training.batch_size,
        shuffle=True,
        num_workers=experiment.data.num_workers,
    )

    # Create torch DataLoader for validation set
    valData = PlantVillage(
        dtype="val",
        data_path=to_absolute_path(paths.data_path),
        process_type=experiment.data.process_type,
    )
    val_loader = valData.get_loader(
        batch_size=experiment.training.batch_size,
        shuffle=False,
        num_workers=experiment.data.num_workers,
    )

    # Initialize model
    model = ImageClassification(
        lr=experiment.training.lr, n_classes=trainData.n_classes
    )
    model.to(device)

    # Train model
    trainer = Trainer(
        max_epochs=experiment.training.epochs,
        accelerator=accelerator_type,
        devices=num_devices,
        logger=WandbLogger(
            name=experiment.experiment_name, project=config.version, entity=loggers.wandb_entity
        ),
    )
    trainer.fit(model, train_loader, val_loader)

    # Determines what information to store in checkpoint
    checkpoint = {
        "configuration": config,
        "state_dict": model.state_dict(),
        "save_time": time.time(),
    }

    # Create folder and save checkpoint/model
    save_path = Path(to_absolute_path(paths.data_path)) / experiment.experiment_name
    os.makedirs(save_path)
    torch.save(checkpoint, save_path / "final.pth")


if __name__ == "__main__":
    train()
