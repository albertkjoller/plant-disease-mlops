import torch
import datetime
from src.models.model import ImageClassification
import time
import os
import datetime
from google.cloud import storage
import ndjson

class ModelWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ = self.load_model("temp/models/trained_model.pth")

        # Create temporary data-folder
        os.makedirs("app/static/assets/models", exist_ok=True)
        os.makedirs("app/static/assets/images", exist_ok=True)

    def load_model(self, torch_filepath):
        try:

            if torch_filepath.split('.')[-1] == 'pth':
                checkpoint = torch.load(torch_filepath)

                # Change load status
                self.loaded = True
                self.filepath = torch_filepath

                # Setup model
                self.model = ImageClassification(
                    lr=checkpoint["training_parameters"]["lr"], n_classes=38
                )
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.to(self.device)
                self.model.eval()

                # Return save-time
                self.load_time = datetime.datetime.fromtimestamp(int(time.time()))
                self.save_time = datetime.datetime.fromtimestamp(
                    int(checkpoint["save_time"])
                )
            elif torch_filepath.split('.')[-1] == 'ckpt':
                self.model = ImageClassification()
                self.model  = self.model.load_from_checkpoint(torch_filepath)
                self.model.to(self.device)
                self.model.eval()
                # Change load status
                self.loaded = True
                self.filepath = torch_filepath
                self.load_time = datetime.datetime.fromtimestamp(int(time.time()))
                self.save_time = (torch_filepath.split('-')[-1]).split('.')[0]

            # Save response
            self.model_response = {
                "loaded": self.loaded,
                "filepath": self.filepath,
                "savetime": self.save_time,
                "loadtime": self.load_time,
            }

        except FileNotFoundError:
            self.model_response = {"loaded": False}

def model_loaded(modelClass):
    try:
        return modelClass.loaded == True
    except:
        return False

def get_base_model(name="epoch=00-val_acc=0.69-13-01-2023 22:45:11.ckpt",trainer=False):
    client = storage.Client("plant-disease-mlops")
    bucket = client.get_bucket("plant-disease-mlops-models")
    blob=bucket.blob(name)
    if not trainer:
        if not os.path.exists(f'app/static/assets/models/test/{name}'):
            blob.download_to_filename(f'app/static/assets/models/test/{name}')
        return name
    else:
        if not os.path.exists(f'models/{name}'):
            blob.download_to_filename(f'models/{name}')
        return f'models/{name}'
