Classification of Plant Diseases using Computer Vision
==============================
- Andreas Theilgaard (s201715), Albert Kjøller Jacobsen (s194253), Phillip C. Højbjerg (s184984)

Course project for 02476 - Machine Learning Operations @ DTU (January 2023)

[![build status](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/ubuntu.yml)
[![build status](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/macos.yml/badge.svg)](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/macos.yml)
[![build status](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/windows.yml/badge.svg)](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/windows.yml)
[![build status](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/coverage.yml/badge.svg)](https://github.com/albertkjoller/plant-disease-mlops/actions/workflows/coverage.yml)

## Project Description

1. **Overall goal of the project:** The goal of this project is to use the PyTorch Image Models [(TIMM)](https://github.com/rwightman/pytorch-image-models) framework for classification of 38 plant diseases on the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset - while complying with the MLOps tools and practices provided in the [course](https://skaftenicki.github.io/dtu_mlops/).

2. **What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics):** The framework used throughout this project will be the PyTorch Image Models [(TIMM)](https://github.com/rwightman/pytorch-image-models) framework.

3. **How do you intend to include the framework into your project:** As the intention of this project is to get acquainted with the MLOps tools provided in the course - and not to develop an all-new AI-model - the framework is used in order to get access to a pretrained ResNet model, that will be used for the classification task at hand.

4. **What data are you going to run on (initially, may change):** The colorised images of the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) Kaggle-dataset, will be used throughout the project. The dataset consists of 54,305 images of healthy and diseased plant-leaves, each labeled with a plant- and disease-identifier. The dataset includes 38 different diseases, of 14 different plant-types. The task is translated to a 38-class classification problem, that identifies the disease, and thereby the plant-type of a given plant-leaf.

5. **What deep learning models do you expect to use:** The project utilizes a pretrained ResNet50 model - first introduced in the original ResNet [paper](https://arxiv.org/abs/1512.03385) (2015). The [code](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py) for the model is available through the TIMM-repository.

---

## Setup

Clone the repository and create a virtual environment (with Python 3.10). A pre-defined environment running with CUDA 11.6 can be created like:

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

### Docker
A Virtual Machine containing all that is needed for training and evaluation of models is available as a Docker image.
It can be accessesed like so:
```
docker pull gcr.io/plant-disease-mlops/docker_img
```

### Pre-commit
For exploiting the `pre-commit`-package when committing to the repository, install with the current configurations by running:
```
pre-commit install
```
Now, this should assist in checking pre-commit hooks when committing new code!

### Data download
After cloning this repository, the download of [data](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) is done through the following command:

```
dvc pull
```

---

### Training locally

In order to train a model locally, an experiment `.yaml-file` should be created/modified, and set as the experiment in the `config.default`-file.
Now, the model can be trained with the following command

```python src/model/train_model.py```

### Running a hyperparameter-sweep
A hyper-parameter sweep can be started through the following command, once the sweep has been defined in `sweep.yaml`:
```
wandb sweep --project sweeps_demo src/configs/sweep.yaml
```
The command will return a `sweep_id`, that starts the sweep like so:
```
wandb agent aap_dtu_mlops/sweeps_demo/sweep_id
```

### Training model on Google Cloud CLI

Accessing the Cloud instance is done through the following command:
```
gcloud compute ssh --zone "europe-west4-a" "plant-disease-mlops-gpu-big-engine"  --project "plant-disease-mlops"
```
Once the instance has been started, the most recent Docker image is started like so:
```
docker run --gpus all -it --rm gcr.io/plant-disease-mlops/docker_img
```
Now, the training can be started in accordance with `Running a hyperparameter-sweep`.

### Gcloud model transfer

```
/root/google-cloud-sdk/bin/gcloud storage cp models/exp1/LR0.0035900486014863666-BS100/-epoch=77-val_acc=0.92.ckpt gs://plant-disease-models
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

---

## Checklist

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github
