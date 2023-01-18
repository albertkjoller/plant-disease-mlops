FROM nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY deployment/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python==4.5.5.64 --no-cache-dir
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir

COPY deployment/app /deployment/app
COPY src/models/model.py /src/models/model.py
COPY src/models/model_utils.py /src/models/model_utils.py

CMD ["uvicorn", "deployment.app.app_setup:app", "--host", "0.0.0.0", "--port", "80"]
