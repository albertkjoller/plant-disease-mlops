import pytest

# from tests import _PATH_DATA
from fastapi import FastAPI
import httpx
from fastapi.testclient import TestClient

#############################

# fix for path, but very ugly
from pathlib import Path
import os
import sys

myDir = os.getcwd()
path = Path(f"{myDir}/app")
a = str(path.parent.absolute())
sys.path.append(a)
from deployment.app.main import router

##############################
from deployment.app.app_setup import create_app

app = create_app()
client = TestClient(app)

# Function 1: '/' Method GET
def test_func1():
    response = client.get("/")
    assert response.status_code == 200

# Function 2: '/upload_model' Method: POST
def test_func2():
    response = client.post("/upload_model")
    assert response.status_code == 200

# Function 3: '/load_model' Method: POST
def test_func3():
    response = client.post("/load_model")
    assert response.json()["model"]["loaded"] == True

# Function 4: '/predict' Method: POST
def test_func4():
    load_model = response = client.post("/load_model")  # start by loading model
    files = {'file': open('example_images/Apple_healthy.jpg','rb')}
    response = client.post("/predict",files=files)
    assert response.json()["output"]["results"]["A"]["0"]["pred"] == 10

def test_func5():
    load_model = response = client.post("/load_model")  # start by loading model
    files = [('files', open('example_images/Apple_healthy.jpg','rb')),
             ('files', open('example_images/Tomato_bacterial.jpg','rb'))]
    response = client.post("/predict_multiple",files=files)
    response = response.json()
    pred1 = response["output"]["results"]["Apple_healthy.jpg"]["0"]["pred"]
    pred2 = response["output"]["results"]["Tomato_bacterial.jpg"]["0"]["pred"]
    assert pred1 == 10 and pred2 == 10