from http import HTTPStatus
from fastapi import Request
from fastapi import UploadFile, File
from fastapi import BackgroundTasks

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from fastapi.responses import HTMLResponse

import os
import glob
from pathlib import Path
from typing import Optional, List

import cv2
import torch
from torchvision import transforms
from PIL import Image

from deployment.app.app_utils import (
    ModelWrapper,
    update_log,
    prepare_feature,
    get_train_distribution,
)

from fastapi.templating import Jinja2Templates
import secrets
from fastapi import APIRouter
import pandas as pd
import numpy as np
import datetime
from csv import writer

router = APIRouter()
templates = Jinja2Templates(directory="./deployment/app/templates")

hash_ = secrets.token_hex(8)
modelClass = ModelWrapper()

# initialize the log file for the prediction data
if not os.path.exists("deployment/app/monitoring/current_data.csv"):
    with open("deployment/app/monitoring/current_data.csv", "w") as file:
        header = ["timestamp", "mean", "std", "min", "max", "Q1", "Q3", "model_path"]
        writer_obj = writer(file)
        writer_obj.writerow(header)


@router.get("/")
def root():
    """Health check."""

    return modelClass.file_upload_result


@router.post("/upload_model")
async def upload_model(file: Optional[UploadFile] = None):
    os.makedirs(f"deployment/app/static/assets/models/{hash_}", exist_ok=True)
    path_ = f"""deployment/app/static/assets/models/{hash_}"""

    if file:
        with open(f"{path_}/" + file.filename, "wb") as f:
            content = await file.read()
            f.write(content)
            f.close()

    response = {
        "upload-successful": True,
        "model": {"filepath": f"{path_}/{file.filename if file else 'default.pth'}"},
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@router.post("/load_model")
def load_model(model_name: Optional[str] = None):
    # Load from temporary storage
    model_name = "default.pth" if model_name == None else f"{hash_}/{model_name}"
    modelClass.load_model(
        f"deployment/app/static/assets/models/{model_name}"
    )  # hash+model_name

    response = {
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@router.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = None,
    h: Optional[int] = 56,
    w: Optional[int] = 56,
):

    os.makedirs(f"deployment/app/static/assets/images/{hash_}", exist_ok=True)
    path_ = f"""deployment/app/static/assets/images/{hash_}"""

    # Store image on server
    with open(f"{path_}/{file.filename}", "wb") as f:
        content = await file.read()
        f.write(content)
        f.close()

    # Load image
    transform=transforms.Compose(
        [
            transforms.Resize(h),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )

    image = Image.open(f"""{path_}/{file.filename}""")
    image = transform(image)
    image = image.unsqueeze(0)

    # Setup input
    input = {"data": image, "label": file.filename.split(os.sep)[-1]}

    # Forward pass through model
    with torch.no_grad():
        batch_idx = -1  # For running in deployment mode
        output = modelClass.model.predict_step(input, batch_idx)
        output_response = {"results": output}

    timestamp = str(datetime.datetime.now())
    features = prepare_feature(image[0])
    background_tasks.add_task(update_log, timestamp, features, modelClass)

    response = {
        "output": output_response,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@router.post("/predict_multiple")
async def predict_multiple(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    h: Optional[int] = 56,
    w: Optional[int] = 56,
):

    images, labels = [], []
    # if modelClass.loaded == True:
    # hash_ = secrets.token_hex(8)
    os.makedirs(f"deployment/app/static/assets/images/{hash_}", exist_ok=True)
    path_ = f"""deployment/app/static/assets/images/{hash_}"""

    for data in files:
        with open(f"""{path_}/{data.filename}""", "wb") as f:
            content = await data.read()
            f.write(content)
            f.close()

        transform=transforms.Compose(
            [
                transforms.Resize(h),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )

        image = Image.open(f"""{path_}/{data.filename}""")
        image = transform(image)

        timestamp = str(datetime.datetime.now())
        features = prepare_feature(image)
        background_tasks.add_task(update_log, timestamp, features, modelClass)

        images.append(image)
        labels.append(data.filename.split(os.sep)[-1])

    input = {"data": torch.stack(images), "label": labels}

    with torch.no_grad():
        batch_idx = -1  # For running in deployment mode
        output = modelClass.model.predict_step(input, batch_idx)
        output_response = {"results": output}

    # else:
    #    output_response = {"results": None}

    response = {
        "output": output_response,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


## Viz endpoint
@router.get("/viz_model_checkpoint")
def checkpoint(request: Request):
    output = templates.TemplateResponse(
        "modelcheckpoint.html",
        context={
            "request": request,
            "file_upload_result": modelClass.file_upload_result,
        },
    )
    return output


@router.post("/viz_model_checkpoint")
async def checkpoint(request: Request, file: UploadFile):

    if file.filename == "":
        checkpoint_name = "default.pth"  # get default model checkpoint
        path_ = "deployment/app/static/assets/models"

    else:
        os.makedirs(f"deployment/app/static/assets/models/{hash_}", exist_ok=True)
        path_ = f"""deployment/app/static/assets/models/{hash_}"""

        with open(f"{path_}/" + file.filename, "wb") as f:
            content = await file.read()
            f.write(content)
            f.close()

        checkpoint_name = file.filename

    modelClass.load_model(f"{path_}/{checkpoint_name}")
    file_upload_result = {
        "upload-successful": True,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    modelClass.file_upload_result = file_upload_result

    return templates.TemplateResponse(
        "modelcheckpoint.html",
        context={"request": request, "file_upload_result": file_upload_result},
    )


@router.get("/viz_model_inference")
def inference(request: Request):
    output = templates.TemplateResponse(
        "inference.html",
        context={
            "request": request,
            "image_upload_result": "",
            "model_loaded": modelClass.loaded,
        },
    )
    return output


@router.post("/viz_model_inference")
async def inference(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = None,
):

    test_ = False
    images, labels = [], []
    w, h = 56, 56
    # if modelClass.loaded == True:
    hash_ = secrets.token_hex(8)
    os.makedirs(f"deployment/app/static/assets/images/{hash_}", exist_ok=True)
    path_ = f"""deployment/app/static/assets/images/{hash_}"""

    for data in files:
        with open(f"""{path_}/{data.filename}""", "wb") as f:
            content = await data.read()
            f.write(content)
            f.close()

        transform=transforms.Compose(
            [
                transforms.Resize(h),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )

        image = Image.open(f"""{path_}/{data.filename}""")
        image = transform(image)

        images.append(image)
        labels.append(
            data.filename.split(os.sep)[-1] if not test_ else data.split(os.sep)[-1]
        )

        timestamp = str(datetime.datetime.now())
        features = prepare_feature(image)
        background_tasks.add_task(update_log, timestamp, features, modelClass)

        input = {"data": torch.stack(images), "label": labels}
        with torch.no_grad():
            batch_idx = -1  # For running in deployment mode
            output = modelClass.model.predict_step(input, batch_idx)
            output_response = {"results": output}

    images = []
    extensions = [".jpg", ".png", ".jpeg",".JPG"]
    for ext in extensions:
        img_paths = glob.glob(Path(f"{path_}/*{ext}").as_posix())
        images += [(os.sep).join(p_.split("/")[3:]) for p_ in img_paths]

    files_list = [img.split(os.sep)[-1] for img in images]

    # else:
    #    output_response = {"results": None}

    image_upload_result = {
        "output": output_response,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }

    output = templates.TemplateResponse(
        "inference.html",
        context={
            "request": request,
            "image_upload_result": image_upload_result,
            "images": images,
            "num_images": len(images),
            "output": output_response,
            "raw_path": files_list,
            "num_predictions": list(range(5)),
            "model_loaded": modelClass.loaded,
        },
    )
    return output


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring():
    reference_data = get_train_distribution()

    current_data = pd.read_csv("deployment/app/monitoring/current_data.csv")
    current_data = current_data[
        current_data["model_path"] == modelClass.model_response["filepath"]
    ].reset_index(drop=True)
    current_data = current_data.drop(columns=["timestamp", "model_path"])

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    data_drift_report.run(current_data=current_data, reference_data=reference_data)
    data_drift_report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
