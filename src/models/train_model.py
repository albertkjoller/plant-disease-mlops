import os
from pathlib import Path

import time
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
import yaml

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.dataloader import PlantVillage
from src.models.model import ImageClassification

import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="defaults.yaml", version_base="1.1")
def train(config):
    """
    Function for training the model
    Attributes
    ----------
        config:
            config file containing configurations for the hyperparameters for training
    """

    print(f"\nRunning on CUDA? {torch.cuda.is_available()}")
    print(f"\nConfiguration: \n {OmegaConf.to_yaml(config)}")

    # Extract information from configuration
    experiment = config.experiment
    paths = config.paths
    loggers = config.logging

    # Set seed
    torch.manual_seed(experiment.training.seed)

    # Define compute scenario
    device, accelerator_type, num_devices = (
        (torch.device("cuda"), "gpu", -1)
        if torch.cuda.is_available()
        else (torch.device("cpu"), "cpu", None)
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
        lr=experiment.training.lr,
        batch_size=experiment.training.batch_size,
        n_classes=trainData.n_classes,
    )
    model.to(device)

    # Define log-name
    log_name = f"{experiment.experiment_name}.time={int(round(time.time()))}.lr={experiment.training.lr}.batch_size={experiment.training.batch_size}.seed={experiment.training.seed}"

    # Define model checkpoint
    save_path = Path(to_absolute_path(paths.save_path)) / experiment.experiment_name
    if not (os.path.exists(save_path) and os.path.isdir(save_path)):
        os.makedirs(save_path)

    # Save the checkpoint for the epoch leading to the best validation accuracy
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_acc",
        mode="max",
        dirpath=save_path,
        filename="{epoch:02d}-{val_acc:.2f}-" + f"{log_name}",
    )

    # Define trainer
    trainer = Trainer(
        max_epochs=experiment.training.epochs,
        accelerator=accelerator_type,
        devices=num_devices,
        logger=WandbLogger(
            name=f"{log_name}-{int(time.time())}",
            project=config.version,
            entity=loggers.wandb_entity,
        ),
        callbacks=[checkpoint_callback],
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
