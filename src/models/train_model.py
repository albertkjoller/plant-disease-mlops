from pytorch_lightning import Trainer
import torch
from src.models.model import ImageClassification
from model import ImageClassification

from src.data.dataloader import PlantVillage

def main():
    device,gpus = (torch.device('cuda'),-1) if torch.cuda.is_available() else (torch.device('cpu'),0)

    #
    dataload = PlantVillage()
    train_loader = dataload.get_loaders(dtype='train',data_path='data/processed/',batch_size=32,shuffle=False)
    val_loader = dataload.get_loaders(dtype='val',data_path='data/processed/',batch_size=32,shuffle=False)
    
    model = ImageClassification()
    model.to(device)

    trainer = Trainer(max_epochs=5,gpus=gpus)
    trainer.fit(model,train_loader,val_loader)
    torch.save(trainer.state_dict(), "models/trained_model.pt")

if __name__ == "__main__":
    main()