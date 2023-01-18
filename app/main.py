import uvicorn
from http import HTTPStatus
from fastapi import FastAPI,Request, Form
from fastapi import UploadFile, File
from fastapi.responses import RedirectResponse

import os
from typing import Optional, List

import cv2
import torch

#from app.app_utils import ModelWrapper,model_loaded
from app_utils import ModelWrapper,model_loaded

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import secrets
from fastapi import APIRouter

#app = create_app()
router = APIRouter()
#router.mount("/static", StaticFiles(directory="./app/static"), name="static")
templates = Jinja2Templates(directory="./app/templates")

modelClass = ModelWrapper()


@router.get("/")
def root():
    """Health check."""

    response = {
        "root": True,
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@router.post("/upload_model")
async def upload_model(file: Optional[UploadFile] = None):

    if not file:
        file = [x for x in os.listdir('app/static/assets/models/test') if x!='.DS_Store'][0]
        test_ = True
        path_ = 'app/static/assets/models/test'
    else:
        test_ = False
        hash_ = secrets.token_hex(8)
        os.makedirs(f"app/static/assets/models/{hash_}", exist_ok=True)
        path_ = f"""app/static/assets/models/{hash_}"""

        with open(f"{path_}/" + file.filename, "wb") as f:
            content = await file.read()
            f.write(content)
            f.close()

    response = {
        "upload-successfull": True,
        "model": {"filepath": file.filename if not test_ else file},
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@router.post("/load_model")
def load_model(model_name: Optional[str]=''):
    # Load from temporary storage
    if model_name:
        modelClass.load_model("app/static/assets/models/" + model_name) # hash+model_name
    else:
        modelClass.load_model("app/static/assets/models/test/epoch=00-val_acc=0.69-13-01-2023 22:45:11.ckpt") # test

    response = {
        "model": modelClass.model_response,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@router.post("/predict")
async def predict(
    file: Optional[UploadFile] = None, h: Optional[int] = 56, w: Optional[int] = 56
):

    test_ = False
    if not file:
        file = [x for x in os.listdir('app/static/assets/images/test') if x!='.DS_Store'][0]
        test_ = True
        path_ = f"""app/static/assets/images/test"""

    if modelClass.loaded == True:
        if not test_:
            hash_ = secrets.token_hex(8)
            os.makedirs(f"app/static/assets/images/{hash_}", exist_ok=True)
            path_ = f"""app/static/assets/images/{hash_}"""

            # Store image on server
            with open(f"app/static/assets/images/{hash_}/" + file.filename, "wb") as f:
                content = await file.read()
                f.write(content)
                f.close()
        # Load image
        image = cv2.imread(f"""{path_}/{file.filename if not test_ else file}""")
        image = cv2.resize(image, (h, w))
        image = torch.FloatTensor(image).view(1, -1, h, w).to(modelClass.device)
        if image.max() > 1.0:
            image /= 256

        # Setup input
        input = {"data": image, "label": file.filename.split(os.sep)[-1] if not test_ else file.split(os.sep)[-1]}

        # Forward pass through model
        with torch.no_grad():
            batch_idx = -1  # For running in deployment mode
            output = modelClass.model.predict_step(input, batch_idx)
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


@router.post("/predict_multiple")
async def predict_multiple(
    files: List[UploadFile] = File(...), h: Optional[int] = 56, w: Optional[int] = 56):

    images, labels = [], []
    if modelClass.loaded == True:
        hash_ = secrets.token_hex(8)
        os.makedirs(f"app/static/assets/images/{hash_}", exist_ok=True)
        path_ = f"""app/static/assets/images/{hash_}"""
        for data in files:
            with open(f"""{path_}/{data.filename}""", "wb") as f:
                content = await data.read()
                f.write(content)
                f.close()

            image = cv2.imread(f"""{path_}/{data.filename}""")
            image = cv2.resize(image, (h, w))
            image = torch.FloatTensor(image).view(-1, h, w).to(modelClass.device)
            if image.max() > 1.0:
                image /= 256

            images.append(image)
            labels.append(data.filename.split(os.sep)[-1])

        input = {"data": torch.stack(images), "label": labels}

        with torch.no_grad():
            batch_idx = -1  # For running in deployment mode
            output = modelClass.model.predict_step(input, batch_idx)
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



## Viz endpoint
@router.get("/viz_model_checkpoint")
def checkpoint(request: Request):
    return templates.TemplateResponse('modelcheckpoint.html', context={'request': request,'file_upload_result':''})


@router.post("/viz_model_checkpoint")
async def checkpoint(request: Request,
                    file: Optional[UploadFile] = None,
                    ):

    if file.filename == '':
        file = [x for x in os.listdir('app/static/assets/models/test') if x!='.DS_Store'][0]
        test_ = True
        path_ = 'app/static/assets/models/test'
    else:
        test_ = False
        hash_ = secrets.token_hex(8)
        os.makedirs(f"app/static/assets/models/{hash_}", exist_ok=True)
        path_ = f"""app/static/assets/models/{hash_}"""

        with open(f"{path_}/" + file.filename, "wb") as f:
            content = await file.read()
            f.write(content)
            f.close()

    file_upload_result = {
        "upload-successfull": True,
        "model": {"filepath": file.filename if not test_ else file},
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    modelClass.load_model(f"{path_}/{file.filename if not test_ else file}")
    return templates.TemplateResponse("modelcheckpoint.html",context={'request':request,'file_upload_result':file_upload_result})

@router.get("/viz_model_inference")
def inference(request: Request):
    return templates.TemplateResponse('inference.html', context={'request': request,'image_upload_result':''})

@router.post("/viz_model_inference")
async def inference(request: Request,
                    files: Optional[List[UploadFile]] = None,
                    ):
    test_ = False
    images, labels = [], []
    w,h=56,56
    if files[0].filename == '':
        files = [x for x in os.listdir('app/static/assets/images/test') if x!='.DS_Store']
        test_ = True
        path_ = f"""app/static/assets/images/test"""
    try:
        if modelClass.loaded == True:
            if not test_:
                hash_ = secrets.token_hex(8)
                os.makedirs(f"app/static/assets/images/{hash_}", exist_ok=True)
                path_ = f"""app/static/assets/images/{hash_}"""

            for data in files:
                if not test_:
                    with open(f"""{path_}/{data.filename}""", "wb") as f:
                        content = await data.read()
                        f.write(content)
                        f.close()
                image = cv2.imread(f"""{path_}/{data.filename if not test_ else data}""")
                image = cv2.resize(image, (h, w))
                image = torch.FloatTensor(image).view(-1, h, w).to(modelClass.device)
                if image.max() > 1.0:
                    image /= 256
                images.append(image)
                labels.append(data.filename.split(os.sep)[-1] if not test_ else data.split(os.sep)[-1])

                input = {"data": torch.stack(images), "label": labels}
                with torch.no_grad():
                    batch_idx = -1  # For running in deployment mode
                    output = modelClass.model.predict_step(input, batch_idx)
                    output_response = {"results": output}
            files_list = [x for x in os.listdir(path_) if x!='.DS_Store']
            images = [f"{path_[11:]}/{x}" for x in files_list]            
        else:
            output_response = {"results": None}

        image_upload_result = {
            "output": output_response,
            "model": modelClass.model_response,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }

        return templates.TemplateResponse("inference.html",context={'request':request,'image_upload_result':image_upload_result,\
            'images':images,'num_images':len(images),\
            'output':output_response,'raw_path':files_list\
            ,'num_predictions':list(range(5)),'model_loaded':model_loaded(modelClass)})
    except:
        return RedirectResponse(url='/viz_model_checkpoint')


#if __name__ == "__main__":
#    # Run application
#    uvicorn.run(app, host="127.0.0.1", port=8000)

