import time
import hydra
from omegaconf import OmegaConf

import torch
from pytorch_lightning import Trainer

from src.data.dataloader import PlantVillage
from src.models.model import ImageClassification

import logging

log = logging.getLogger(__name__)


#@hydra.main(config_path="config", config_name='default_config.yaml')
#def train(config):
#    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
#    hparams = config.experiment

def train():    
    # Define compute scenario 
    device, gpus = (
        (torch.device("cuda"), -1)
        if torch.cuda.is_available()
        else (torch.device("cpu"), 0)
    )

    # Define run-specific parameters
    SAVE_PATH = 'models/trained_model.pth'
    DATA_PATH = 'data/processed'
    PROCESS_TYPE = 'color'
    LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 2

    # Create torch DataLoader for training set
    trainData = PlantVillage(
        dtype='train', data_path=DATA_PATH, process_type=PROCESS_TYPE,
    )
    train_loader = trainData.get_loader(
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    # Create torch DataLoader for validation set
    valData = PlantVillage(
        dtype='val', data_path=DATA_PATH, process_type=PROCESS_TYPE,
    )
    val_loader = valData.get_loader(
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )

    # Initialize model
    model = ImageClassification(lr=LR, n_classes=trainData.n_classes)
    model.to(device)

    # Train model
    trainer = Trainer(max_epochs=EPOCHS, gpus=gpus)
    trainer.fit(model, train_loader, val_loader)

    # Save model
    checkpoint = {
        "training_parameters": {
            "data_path": DATA_PATH,
            "save_path": SAVE_PATH,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": device,
        },
        "state_dict": model.state_dict(),
        "save_time": time.time(),
    }
    torch.save(checkpoint, SAVE_PATH)

if __name__ == "__main__":
    train()
