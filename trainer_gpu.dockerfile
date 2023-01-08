# Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy setup and installation files 
COPY requirements.txt requirements.txt
COPY setup.py setup.py
# copy data
COPY data/processed/color/train data/processed/color/train
COPY data/processed/color/val data/processed/color/val
# copy model training files
COPY src/configs src/configs
COPY src/data/dataloader.py src/data/dataloader.py
COPY src/models/model.py src/models/model.py
COPY src/models/train_model.py src/models/train_model.pyc

WORKDIR /
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
