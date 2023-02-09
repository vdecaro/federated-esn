FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN pip install -U "ray[tune]"
WORKDIR /app

ENTRYPOINT [ "python", "main.py" ]