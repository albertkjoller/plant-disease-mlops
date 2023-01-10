import pytest
import os
from pathlib import Path

from tests import _PATH_DATA
from src.data.dataloader import PlantVillage    
        

@pytest.mark.skipif(not os.path.exists(Path(_PATH_DATA) / 'processed'), reason="Data files not found")
def test_dataset_size():
    # Create train dataset 
    trainData = PlantVillage(
        dtype="train",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create validation dataset 
    valData = PlantVillage(
        dtype="val",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create test dataset 
    testData = PlantVillage(
        dtype="test",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    
    # Total number of points in processed data
    N = trainData.N_points + valData.N_points + testData.N_points
    
    assert N                    == 54305, "The number of data points in the processed datafiles differ from the size of the raw dataset..."
    assert trainData.N_points   == 34755, "Size of training data does not correspond to specifications from ./src/data/make_dataset.py"
    assert valData.N_points     == 8689, "Size of validation data does not correspond to specifications from ./src/data/make_dataset.py"
    assert testData.N_points    == 10861, "Size of test data does not correspond to specifications from ./src/data/make_dataset.py"

@pytest.mark.skipif(not os.path.exists(Path(_PATH_DATA) / 'processed'), reason="Data files not found")
def test_datapoints_shape():
    # Create train dataset 
    trainData = PlantVillage(
        dtype="train",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create validation dataset 
    valData = PlantVillage(
        dtype="val",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create test dataset 
    testData = PlantVillage(
        dtype="test",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )

    assert trainData.images.shape == (34755, 3, 56, 56), "Train data does not have correct shape"
    assert valData.images.shape == (8689, 3, 56, 56), "Validation data does not have correct shape"
    assert testData.images.shape == (10861, 3, 56, 56), "Test data does not have correct shape"

@pytest.mark.skipif(not os.path.exists(Path(_PATH_DATA) / 'processed'), reason="Data files not found")
def test_label_presence():
    # Create train dataset 
    trainData = PlantVillage(
        dtype="train",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create validation dataset 
    valData = PlantVillage(
        dtype="val",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create test dataset 
    testData = PlantVillage(
        dtype="test",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )

    assert all([i in trainData.labels for i in range(38)]), "Not all labels occur in the training set..."
    assert all([i in valData.labels for i in range(38)]), "Not all labels occur in the training set..."
    assert all([i in testData.labels for i in range(38)]), "Not all labels occur in the training set..."

"""
@pytest.mark.skipif(not os.path.exists(Path(_PATH_DATA) / 'processed'), reason="Data files not found")
class TestDatasetClass:
    global trainData, valData, testData

    # Create train dataset 
    trainData = PlantVillage(
        dtype="train",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create validation dataset 
    valData = PlantVillage(
        dtype="val",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )
    # Create test dataset 
    testData = PlantVillage(
        dtype="test",
        data_path=Path(_PATH_DATA) / 'processed',
        process_type='color',
    )

    def test_dataset_size(self):
        # Total number of points in processed data
        N = trainData.N_points + valData.N_points + testData.N_points
        
        assert N                    == 54305, "The number of data points in the processed datafiles differ from the size of the raw dataset..."
        assert trainData.N_points   == 34755, "Size of training data does not correspond to specifications from ./src/data/make_dataset.py"
        assert valData.N_points     == 8689, "Size of validation data does not correspond to specifications from ./src/data/make_dataset.py"
        assert testData.N_points    == 10861, "Size of test data does not correspond to specifications from ./src/data/make_dataset.py"

    def test_datapoints_shape(self):
        assert trainData.images.shape == (34755, 3, 56, 56), "Train data does not have correct shape"
        assert valData.images.shape == (8689, 3, 56, 56), "Validation data does not have correct shape"
        assert testData.images.shape == (10861, 3, 56, 56), "Test data does not have correct shape"

    def test_label_presence(self):
        assert all([i in trainData.labels for i in range(38)]), "Not all labels occur in the training set..."
        assert all([i in valData.labels for i in range(38)]), "Not all labels occur in the training set..."
        assert all([i in testData.labels for i in range(38)]), "Not all labels occur in the training set..."
"""