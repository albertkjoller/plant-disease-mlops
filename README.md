Classification of Plant Diseases using Computer Vision
==============================
- Andreas Theilgaard (s201715), Albert Kjøller Jacobsen (s194253), Phillip C. Højbjerg (s184984)

Course project for 02476 MLOps

1. **Overall goal of the project:** The goal of this project is to use the PyTorch Image Models [(TIMM)](https://github.com/rwightman/pytorch-image-models) framework for classification of 38 plant diseases on the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset - while complying with the MLOps tools and practices provided in the [course](https://skaftenicki.github.io/dtu_mlops/). 

2. **What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics):** The framework used throughout this project will be the PyTorch Image Models [(TIMM)](https://github.com/rwightman/pytorch-image-models) framework. 

3. **How do you intend to include the framework into your project:** As the intention of this project is to get acquainted with the MLOps tools provided in the course - and not to develop an all-new AI-model - the framework is used in order to get access to a pretrained ResNet model, that will be used for the classification task at hand. 

4. **What data are you going to run on (initially, may change):** The colorised images of the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) Kaggle-dataset, will be used throughout the project. The dataset consists of 54,305 images of healthy and diseased plant-leaves, each labeled with a plant- and disease-identifier. The dataset includes 38 different diseases, of 14 different plant-types. The task is translated to a 38-class classification problem, that identifies the disease, and thereby the plant-type of a given plant-leaf. 

5. **What deep learning models do you expect to use:** The project utilizes a pretrained ResNet50 model - first introduced in the original ResNet [paper](https://arxiv.org/abs/1512.03385) (2015). The [code](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py) for the model is available through the TIMM-repository.

## Setup

Create a virtual environment (with Python 3.10). A pre-defined environment running with CUDA 11.6 can be created like:

### Precompiled environment
```
    conda env create -f environment.yml
```

### Manual installation
Otherwise, run the following:

```
    conda create -n mlops_project python=3.10
```

Install the dependencies:
```
    pip install -r requirements.txt
```

#### PyTorch - CPU
If running on CPU install Pytorch with the following command:

```
    pip3 install torch torchvision torchaudio
```

#### PyTorch - GPU (CUDA 11.6)
If running on GPU with CUDA 11.6 install Pytorch with the following command:
```
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Data download
After cloning this repository, the download of [data](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) is done through the following command:

```
    dvc pull
```


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Checklist

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages (using conda)
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use wandb to log training progress and other important metrics/artifacts in your code
* [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] (optional) Create a new project on `gcp` and invite all group members to it
* [ ] Create a data storage on `gcp` for you data
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training on `gcp`
* [ ] Play around with distributed data loading
* [ ] (optional) Play around with distributed model training
* [ ] Play around with quantization and compilation for you trained models

### Week 3

* [ ] Deployed your model locally using TorchServe
* [ ] Checked how robust your model is towards data drifting
* [ ] Deployed your model using `gcp`
* [ ] Monitored the system of your deployed model
* [ ] Monitored the performance of your deployed model

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Create a presentation explaining your project
* [ ] Uploaded all your code to github
* [ ] (extra) Implemented pre*commit hooks for your project repository
* [ ] (extra) Used Optuna to run hyperparameter optimization on your model
