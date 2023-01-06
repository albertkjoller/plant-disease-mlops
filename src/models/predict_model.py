###### Currently not updated

# from pytorch_lightning import Trainer
# import torch
# from src.models import ImageClassification
# #from src.data import DataModule

# def main():
#     device,gpus = (torch.device('cuda'),-1) if torch.cuda.is_available() else (torch.device('cpu'),0)

#     #
#     data = DataModule()

#     #
#     model = ImageClassification()
#     model.to(device)

#     #
#     trainer = Trainer(max_epochs=5,gpus=gpus)
#     trainer.fit(model,data_train,data_val)
#     torch.save(trainer.state_dict(), "models/trained_model.pt")


# if __name__ == "__main__":
#     main()
