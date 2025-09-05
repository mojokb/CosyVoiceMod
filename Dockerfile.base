# CosyVoice Base Image - RTX 3090 Optimized
# 안정적인 시스템/파이썬/파이토치 스택만 포함 (변경 드문 것들)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---- Env ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.6" \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

WORKDIR /app

# ---- System deps (lean) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget unzip zip tar \
    build-essential cmake pkg-config gcc g++ \
    git git-lfs \
    ffmpeg libsndfile1-dev sox libsox-fmt-all \
    libasound2-dev portaudio19-dev \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python aliases (ubuntu 22.04 기본 python3.10이지만, 명확히 고정)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip toolchain
RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Git LFS
RUN git lfs install --system

# ---- PyTorch (CUDA 12.1) ----
RUN python -m pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Core ML/Audio libs (stable) ----
RUN python -m pip install --no-cache-dir \
    numpy==1.24.4 scipy==1.11.4 \
    librosa==0.10.2 soundfile==0.12.1 \
    transformers==4.51.3 \
    Pillow==10.3.0 \
    tqdm==4.66.4 requests==2.31.0 pyyaml==6.0.1 \
    rich==13.7.1

# Whisper (STT)
RUN python -m pip install --no-cache-dir openai-whisper==20231117

# Gradio (UI)
RUN python -m pip install --no-cache-dir gradio==5.4.0

# Model hubs
RUN python -m pip install --no-cache-dir \
    huggingface-hub==0.23.0 modelscope==1.20.0

# (옵션) ONNX / TensorRT는 기본 제외 — 필요 시 앱 레이어에서 추가 권장
# 예: python -m pip install --no-cache-dir onnx onnxruntime-gpu --extra-index-url <cuda12 index>
# 예: python -m pip install --no-cache-dir tensorrt-cu12*  # 시스템 종속성 확인 필요

# ---- Dirs & perms ----
RUN mkdir -p /app/.cache /app/pretrained_models /tmp && \
    chmod 755 /app /app/.cache /app/pretrained_models

# ---- system_info.sh (간단 & 안전) ----
RUN printf '%s\n' '#!/bin/bash' \
'echo "🐳 CosyVoice Base Image Info"' \
'echo "=========================="' \
'grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d \"' \
'echo -n "🐍 Python: "; python --version' \
'echo -n "🔥 PyTorch: "; python -c "import torch; print(torch.__version__)"' \
'echo -n "🎮 CUDA Available: "; python -c "import torch; print(torch.cuda.is_available())"' \
'if command -v nvidia-smi >/dev/null 2>&1; then' \
'  echo "🎯 GPU Info:"' \
'  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits' \
'fi' \
'echo "📦 Key Packages:"' \
'python -m pip show torch torchvision torchaudio gradio openai-whisper transformers | grep -E "Name:|Version:"' \
'echo "=========================="' \
> /app/system_info.sh && chmod +x /app/system_info.sh

# 베이스 마커
RUN echo "CosyVoice Base Image v2.0-rtx3090" > /app/.base_image_info

# Labels
LABEL maintainer="CosyVoice Team" \
      version="2.0-base-rtx3090" \
      description="CosyVoice Base Image with PyTorch/cu121 and audio libs for RTX 3090" \
      gpu.architecture="ampere" \
      gpu.compute="8.6" \
      base.image="true" \
      cuda.version="12.1" \
      python.version="3.10" \
      pytorch.version="2.3.1"

CMD ["/app/system_info.sh"]

