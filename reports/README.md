---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
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
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [x] Check how robust your model is towards data drifting
* [x] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [x] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group number: 6

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s201715, s194253, s184984

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the Pytorch Image Models (Timm) framework throughout our project, in order to load a Pre-trained ResNet50 model. The model was loaded through the following command, ` self.model = timm.create_model( "resnet50", pretrained=True, num_classes=n_classes ) `.

The model was incredibly simple to load, with the only required input on our part was the amount of classes in our plant-disease problem, 38. This made things easier for us, as we could then spend more time on implementing the MLOps tools, rather than developing and training a big model.

Once the model had been loaded, we froze all the weights. Thus making sure not to alter the already useful layers of the ResNet, and instead making our training of the model focus only on the output layer. - We also made sure to add log-softmax, for numerical stability.

The Timm-framework was a great help in the development of this project, although not much time was spent with it.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

In this project, we've emphasized the importance of easy installment for team-members and other potential researchers that would want to use our developed project in the future. Therefore, we've spent time making a README.md that is easy to understand, and contains all the lines of code needed to start the project, as well as all other relevant lines of code for training, predicting, etc.

The following is a snippet of our README.md, explaining the setup of our environment.
---

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

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

We used the cookiecutter format for initializing the project's file structure. However, after initializing with cookiecutter we have
modified the code structure a bit to fit our needs. First of all, we removed the `external` and `interim` folders within the `data` folder as the raw data as well as a processed version is sufficient for us. As we are working from a CLI, we removed the `notebooks` folder as well. The folders `references`, `docs`, `src.features` and `src.visualization` was deemed irrelevant - the latter two have implicitly been implemented in other scripts. We additionally created a `tests`-folder for running pytests as well as a `deployment` folder containing the FastAPI application and the associated Dockerfile for building the deployment-image. Additionally files determingin the pre-commit-configuration, the GCP Trigger (cloudbuild.yaml) as well as modified configurations for flake8, etc. were included in the root of the repository. Furthermore, scripts for creating a PyTorch dataloader as well as the PyTorch-Lightning-based models were included in `src.models` and `src.data`, respectively. We also created the folder `src.configs` for managing configurations with Hydra.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- question 6 fill here ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 9 tests. 3 tests concern the data, where it is tested if correct dimensions and size of data are encountered. 5 tests are used for testing the behaviour of API endpoints, for instance if the model upload or prediction of images endpoint works as expected. Finally, 1 test for testing the model is implemented, where it is tested that the model output is as expected.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage percentage is 48% which is based on all code in the repository (source code, deployment code and tests code) - we refer to [this figure](figures/coverage.png) for seeing the specifics about files and lines not covered within the `pytest`s. We are far from 100% coverage for several reasons - first of all we put more weights on other parts of the project such as the hyperparameter sweeping with `wandb`. Secondly, the data-dependent pytests are skipped as the data was not included into the GitHub actions. As can be seen on the refered figure, it is especially the model.py file and the main.py (deployment/app-folder) that suffers from low coverage - this is due to not being able to train nor test the visual part of the FastAPI app.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Both branches and pull requests (PRs) were used throughout the project. Each time a new code section/feature/improvement a new branch was created, where the work then was developed. When the development had been done, a pull request was made, where at least one other group member had to review the changes. Also an internal rule of all unit test needed to pass was made before a merge into main could take place. Most of the times only a single team member worked on a given branch. However, occasionally two group members were working on different tasks related to one branch, as such a sub-branch was made where one team member worked on the assigned task and another on the branch. When the task on the sub-branch had been developed a merge into the branch was made, and when all the tasks related to this branch had been devloped, and it was ensured that this branch was working, a merge into the main branch was made. As such merge into the main branch only occurred when the entire task was done and not just sub-tasks. Overall the usage of branches and pull requests was a great help to ensure version control and it helped us to manage and maintain code quality.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We made use of DVC in order to push data to a Google Cloud storage bucket. However, once we tried to pull our data to our Docker image using `dvc pull` in the Dockerfile, we seemed to encounter multiple problems - as we didn't want to clone our entire Git repository into our image in order to save space.

Instead, we ended up downloading the data to our Docker images through a gsutil-command, that would copy the data from the bucket into our Docker Image.
```
gsutil cp -r gs://plant-disease-mlops-data-bucket .
```
This did not mean we did not use DVC, as it still helped greatly as a help to continuously push the newest data to the bucket, able to be accessed through gsutil.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

Our Continuous Integration pipeline made use of unit-testing, github actions, pre-commits as well as continous Docker containers.

While we didn't go into great dephts regarding unit-testing, we still wanted to make sure that we understood how to create them and make sure they were running. We created tests regarding the data-part of our pipeline, the model-part as well as the API-part.

- The unit-tests regarding our data made sure that the size and shape of our train-, test- and validation-sets were as expected, as well as making sure that all three sets contained all 38 labels.
- The model-tests made sure that the output of our model, based on a dummy-input, was of correct shape, namely 38 classes.
- Finally, the API-tests made sure our webpage gave correct responses based on various requests to our website.

Our Github actions workflow included a coverage test, as well as tests on 3 OSes (Windows, Ubuntu, MacOS) as seen in the following [https://github.com/albertkjoller/plant-disease-mlops/tree/main/.github/workflows](Workflow-link). Each OS-test ran on both Python 3.9 and 3.10.

Meanwhile, we created pre-commits that would check (and fix) various parts of our pushed code (as already answered in question 6)

Finally, the Trigger-tab on Google Cloud allowed continuous integration in the form of automatic Docker images, that would be pushed to the Google Cloud container registry once built. The trigger would start whenever a push was made to our main branch, a protected branch that required approvals from other team-members.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We made use of Hydra when configuring our experiments, that way we could easily create configuration files as well as easily overwrite them when running in the terminal.

Running an experiment would be done through the following command from the root of the project directory:

```python src/models/train_model.py experiment.training.lr=0.01 experiment.training.batch_size=32```

Meanwhile, a WandB sweep.yaml file was created, that could easliy overwrite the Hydra config through the following command:

```wandb sweep --project sweeps_demo src/configs/sweep.yaml```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- question 13 fill here ---
ANSWER ME!

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

As is seen on [this figure](figures/wandb_sweep.png), we created an experiment in "Weights & Bias" by varying the learning rate associated with model training as well as the batch size used within the data loader. This was done by creating a hyperparameter sweep using a Bayesian Optimization strategy - thereby exploiting hyperparameters that were determined to make the most impact based on values and performance results previously obtained in the sweep-run. As can be seen on the figure, we have chosen to track the validation loss, validation accuracy as well as the training loss and training accuracy where the validation loss was used for optimization and hyperparameter selection. The validation loss was a NLLLoss based on log-softmax outputs for numerical stabilities and determines the data fit whereas the accuracy was used for obtaining a more intuitive representation of how the trained model performs. It is important to track the training loss and accuracy as well for being able to determine whether the model is prone to overfitting - overfitting was not observed. As can be seen on the figures, sweeping the hyperparameters had a significant effect on the model performance on the validation data set with a convergent accuracy ranging between 80% and 92%. Furthermore, we see that the plateau of convergence is dependent on the hyperparameter value - most likely the learning rate.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We faced a variety of bugs in the process of carying out the project. First of all we had code-related bugs - these were directly handled using the debugging tool of Visual Studio Code. For some scripts - such as for running the `make_dataset.py` as well as for training the model in `train_model.py` with a specific experiment configuration - debugging with input arguments was required. In Visual Studio Code this was handled by adding these to the "args" key in  the run-configuration before running in debug mode. For ML-related bugs (such as shape errors and weird model performance (such as always predicting the same class)) the VSCode debugging tool was similarly used by running with breakpoints and interacting in the debug console. Though we do not think that the code is "already perfect", we did not run with a profiler as we kept the model training loop as well as the inference step rather simple by exploiting a pretrained ResNet model with frozen weights.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following GCP services:
1. Engine
2. Bucket
3. Cloud Run
The engine is used for running code on google servers. We used the engine for running model experiments. This was especially useful when doing different sweeps via Weights & Biases using bayesian optimization in order to find the best hyperparameters. This process would not have been posible timewise using a CPU. However, using the google engine with a GPU configuration it speeded up the process a lot.

We uses the bucket for storing data. We both store the plant disease dataset in the bucket and files used for the fastapi application. This ensures easy access to the data.

Finally, we use the cloud run service to serve and run our fastapi application.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

The main GCP compute engine with the following hardware configurations was used.
* GPU:  NVIDIA V100
* CPU: n1-standard-4 (15 GB memory)
* Storage: 50 GB

The engine was started using the following command
```
gcloud compute ssh --zone "europe-west4-a" "plant-disease-mlops-gpu-big-engine"  --project "plant-disease-mlops"
```

The engine was especially useful when doing our experiments, where it was used for fast and efficient computations. The extra disks of 50 GB memory was added since we ran into memory problems when creating the docker images, and adding the extra memory helped solving this issue.

However, when we were testing the setup for the image containers a smaller GCP compute instance was used.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

For deployment we utilized the fastapi framework in order to serve our plant disease model. The model was first deployed locally by first a .py file and then the api was wrapped onto docker container and was ran locally there. When it was ensured these steps worked, the docker container was pushed to gcp and finally deployed via cloud run. Our deployed model comes with both a backend and frontend user interface with the same functionalties in both. To use the api the following command can for instance be called.

* ```curl -X 'POST' \
  'https://plant-disease-fastapi-2c2zw42era-ew.a.run.app/predict?h=56&w=56' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.JPG;type=image/jpeg
  ```

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We managed to implement some monitoring of our deployed model. For instance, we continuously monitor the uploaded data. Each time a new data point is uploaded for prediction some features based on the image are created which are stored in a log file. This log file can be compared against the data distribution on which the model was trained upon via Evidently AI using the api endpoint `monitoring`. Thus, we can continuously track if data drifting is taken place. This information is valuable, as when data drifting occurs the model accuracy might drop, as such a new model including the drifted data can be trained which hopefully should increase the accuracy once again. However, as such re-training loop has not been implemented. We only monitor if data drifting seems to be taken place or not.


### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
