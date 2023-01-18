import torch
import datetime
from http import HTTPStatus

from src.models.model import ImageClassification
import time
import os
import datetime
from csv import writer
from PIL import Image, ImageEnhance
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


class ModelWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ = self.load_model("deployment/app/static/assets/models/default.pth")
        self.loaded = True

        self.file_upload_result = {
            "upload-successful": True,
            "model": self.model_response,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }

        # Create temporary data-folder
        os.makedirs("deployment/app/static/assets/models", exist_ok=True)
        os.makedirs("deployment/app/static/assets/images", exist_ok=True)

    def load_model(self, torch_filepath):
        try:
            if torch_filepath.split(".")[-1] == "pth":
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

            elif torch_filepath.split(".")[-1] == "ckpt":
                self.model = ImageClassification()
                self.model = self.model.load_from_checkpoint(torch_filepath)
                self.model.to(self.device)
                self.model.eval()
                # Change load status
                self.loaded = True
                self.filepath = torch_filepath
                self.load_time = datetime.datetime.fromtimestamp(int(time.time()))
                self.save_time = (torch_filepath.split("-")[-1]).split(".")[0]

            # Save response
            self.model_response = {
                "loaded": self.loaded,
                "filepath": self.filepath,
                "savetime": self.save_time,
                "loadtime": self.load_time,
            }

        except FileNotFoundError:
            self.model_response = {"loaded": False}


def update_log(timestamp: str, features, modelClass):
    mu = np.mean(features)
    sigma = np.std(features)
    Q0 = np.min(features)
    Q4 = np.max(features)
    Q1 = np.quantile(features, q=0.25)
    Q3 = np.quantile(features, q=0.75)
    model_path = modelClass.model_response["filepath"]
    row = [timestamp, mu, sigma, Q0, Q4, Q1, Q3, model_path]
    with open("deployment/app/monitoring/current_data.csv", "a") as file:
        writer_obj = writer(file)
        writer_obj.writerow(row)


def prepare_feature(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    img_features = model.get_image_features(inputs["pixel_values"])
    features_np = img_features.detach().numpy()[0]
    return features_np


def get_train_distribution():
    return pd.read_csv("deployment/app/monitoring/reference_data.csv")
