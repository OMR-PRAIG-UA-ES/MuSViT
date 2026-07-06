FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install imagemagick -y
RUN apt install curl -y
RUN apt clean

RUN pip install uv
RUN pip install huggingface_hub

WORKDIR /workspace

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-install-project

COPY . .
RUN uv sync