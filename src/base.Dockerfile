FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY requirements.txt /app/requirements.txt

WORKDIR /app/

RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt