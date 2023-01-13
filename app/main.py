from fastapi import FastAPI

import re
from http import HTTPStatus

import uvicorn

from enum import Enum
from pydantic import BaseModel

import os
from pathlib import Path

from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import cv2
from typing import Optional, List

import torch
import sys
import datetime, time

from src.models.model import ImageClassification
from app.app_utils import get_labels # get labels dictionary here


class ModelWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ = self.load_model("temp/models/trained_model.pth")

        # Create temporary data-folder
        os.makedirs("temp/data", exist_ok=True)
        os.makedirs("temp/models", exist_ok=True)

    def load_model(self, torch_filepath):
        try:
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

            # Save response
            self.model_response = {
                "loaded": self.loaded,
                "filepath": self.filepath,
                "savetime": self.save_time,
                "loadtime": self.load_time,
            }

        except FileNotFoundError:
            self.model_response = {"loaded": False}


app = FastAPI()
modelClass = ModelWrapper()


@app.get("/")
def root():
    """Health check."""

    response = {
        "root": True,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    # Store model locally
    with open("temp/models/" + file.filename, "wb") as f:
        content = await file.read()
        f.write(content)
        f.close()

    response = {
        "upload-successfull": True,
        "model": {"filepath": file.filename},
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/load_model")
def load_model(model_name: str):
    # Load from temporary storage
    modelClass.load_model("temp/models/" + model_name)

    response = {
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), h: Optional[int] = 56, w: Optional[int] = 56
):

    if modelClass.loaded == True:
        # Store image on server
        with open("temp/data/" + file.filename, "wb") as f:
            content = await file.read()
            f.write(content)
            f.close()

        # Load image
        image = cv2.imread("temp/data/" + file.filename)
        image = cv2.resize(image, (h, w))
        image = torch.FloatTensor(image).view(1, -1, h, w).to(modelClass.device)
        if image.max() > 1.0:
            image /= 256

        # Setup input
        input = {"data": image, "label": file.filename.split(os.sep)[-1]}

        # Forward pass through model
        with torch.no_grad():
            output = modelClass.model.predict_step(input, 0)
            output_response = {"results": output}

    else:
        output_response = {"results": None}

    response = {
        "output": output_response,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict_multiple/")
async def predict_multiple(
    files: List[UploadFile] = File(...), h: Optional[int] = 56, w: Optional[int] = 56
):

    images, labels = [], []
    if modelClass.loaded == True:
        for data in files:
            print("\n\n\n")
            print(data.filename)
            print("\n\n\n")

            with open("temp/data/" + data.filename, "wb") as f:
                content = await data.read()
                f.write(content)
                f.close()

            image = cv2.imread("temp/data/" + data.filename)
            image = cv2.resize(image, (h, w))
            image = torch.FloatTensor(image).view(-1, h, w).to(modelClass.device)
            if image.max() > 1.0:
                image /= 256

            images.append(image)
            labels.append(data.filename.split(os.sep)[-1])

        input = {"data": torch.stack(images), "label": labels}

        with torch.no_grad():
            output = modelClass.model.predict_step(input, 0)
            output_response = {"results": output}

    else:
        output_response = {"results": None}

    response = {
        "output": output_response,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


if __name__ == "__main__":
    # Run application
    uvicorn.run(app, host="127.0.0.1", port=8000)
