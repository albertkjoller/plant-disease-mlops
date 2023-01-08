# Base image
FROM python:3.10-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy files 
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/processed data/processed

WORKDIR /
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
#RUN pip3 install torch torchvision torchaudio
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]