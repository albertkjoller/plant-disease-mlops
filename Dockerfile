# Base image
FROM python:3.10-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN curl https://sdk.cloud.google.com > install.sh
RUN bash install.sh --disable-prompts

#WORKDIR /bin/
SHELL ["/bin/bash", "-c"]
# RUN source /root/google-cloud-sdk/completion.bash.inc
#RUN source /root/google-cloud-sdk/path.bash.inc
COPY DATA_KEY_FILE.json DATA_KEY_FILE.json
RUN /root/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file=DATA_KEY_FILE.json
RUN /root/google-cloud-sdk/bin/gsutil cp -r gs://plant-disease-mlops-data-bucket .

RUN mv plant-disease-mlops-data-bucket/data data
RUN rm -r plant-disease-mlops-data-bucket

# RUN rm /bin/sh && ln -s /bin/bash /bin/sh
#RUN gcloud init

#RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
#RUN apt-get install apt-transport-https ca-certificates gnupg curl -y
#RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
#RUN apt-get update && apt-get install google-cloud-sdk -y


# copy setup and installation files
COPY src/configs src/configs
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# copy data
#COPY data/processed/color/train data/processed/color/train
#COPY data/processed/color/val data/processed/color/val
# copy model training files
COPY src/data/dataloader.py src/data/dataloader.py
COPY src/models/model.py src/models/model.py
COPY src/models/train_model.py src/models/train_model.py

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip3 install torch torchvision torchaudio
