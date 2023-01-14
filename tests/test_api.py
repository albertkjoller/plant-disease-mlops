
import pytest
#from tests import _PATH_DATA
from fastapi import FastAPI
import httpx
from fastapi.testclient import TestClient
from app.app_setup import create_app

app = create_app()
client = TestClient(app)

# Function 1: '/' Method GET
def func1(client):
    response = client.get('/')
    assert response.status_code == 200

# Function 2: '/upload_model' Method: POST
def func2(client):
    response = client.post('/upload_model')
    assert response.status_code == 200

# Function 3: '/load_model' Method: POST
def func3(client):
    response = client.post('/load_model')
    assert response.json()['model']['loaded'] == True

# Function 4: '/predict' Method: POST
def func4(client):
    load_model=response = client.post('/load_model') # start by loading model
    response = client.post('/predict')
    assert response.json()['output']['results']['A']['0']['pred']==15
