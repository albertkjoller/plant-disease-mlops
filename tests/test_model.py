import torch
from src.models.model import ImageClassification

import pytest
from tests import _PATH_DATA

lr = 1e-3
n_classes = 38

LightningModel = ImageClassification(lr=lr, n_classes=n_classes)

def test_model_output():
    dummy_input = torch.randn(1, 3, 56, 56)
    output = LightningModel.model(dummy_input)
    assert output.flatten().shape[0] == n_classes