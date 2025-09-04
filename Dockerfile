# CosyVoice Gradio Docker Image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 설치
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본 Python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# pip 설치
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# CosyVoice 의존성 설치 (PyTorch, Gradio 등 모든 필요 패키지 포함)
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# git lfs 설치 (모델 다운로드용)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs && \
    git lfs install

# 애플리케이션 코드 복사
COPY . /app/

# third_party 서브모듈이 있는 경우를 위한 초기화
RUN if [ -f .gitmodules ]; then \
        git submodule update --init --recursive || echo "No submodules or failed to initialize"; \
    fi

# Matcha-TTS 경로 설정을 위한 심볼릭 링크 (서브모듈이 없는 경우 대비)
RUN mkdir -p third_party/Matcha-TTS || true

# 모델 다운로드 스크립트 생성
RUN echo '#!/bin/bash\n\
echo "Downloading CosyVoice models..."\n\
mkdir -p /app/pretrained_models\n\
\n\
# CosyVoice2-0.5B 다운로드\n\
if [ ! -d "/app/pretrained_models/CosyVoice2-0.5B" ]; then\n\
    echo "Downloading CosyVoice2-0.5B..."\n\
    python -c "from modelscope import snapshot_download; snapshot_download('"'"'iic/CosyVoice2-0.5B'"'"', local_dir='"'"'/app/pretrained_models/CosyVoice2-0.5B'"'"')"\n\
fi\n\
\n\
# CosyVoice-300M-SFT 다운로드\n\
if [ ! -d "/app/pretrained_models/CosyVoice-300M-SFT" ]; then\n\
    echo "Downloading CosyVoice-300M-SFT..."\n\
    python -c "from modelscope import snapshot_download; snapshot_download('"'"'iic/CosyVoice-300M-SFT'"'"', local_dir='"'"'/app/pretrained_models/CosyVoice-300M-SFT'"'"')"\n\
fi\n\
\n\
# CosyVoice-ttsfrd 다운로드\n\
if [ ! -d "/app/pretrained_models/CosyVoice-ttsfrd" ]; then\n\
    echo "Downloading CosyVoice-ttsfrd..."\n\
    python -c "from modelscope import snapshot_download; snapshot_download('"'"'iic/CosyVoice-ttsfrd'"'"', local_dir='"'"'/app/pretrained_models/CosyVoice-ttsfrd'"'"')"\n\
fi\n\
\n\
echo "Model download completed!"\n' > /app/download_models.sh

RUN chmod +x /app/download_models.sh

# 컨테이너 시작 스크립트 생성
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🎙️ CosyVoice Docker Container Starting..."\n\
\n\
# GPU 확인\n\
if nvidia-smi > /dev/null 2>&1; then\n\
    echo "✅ NVIDIA GPU detected"\n\
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits\n\
else\n\
    echo "⚠️ No GPU detected, running on CPU"\n\
fi\n\
\n\
# 모델 다운로드\n\
echo "📥 Checking and downloading models if needed..."\n\
/app/download_models.sh\n\
\n\
# Gradio 앱 실행\n\
echo "🚀 Starting Gradio web app..."\n\
echo "🌐 Web interface will be available at: http://localhost:7860"\n\
python /app/gradio_app.py\n' > /app/start.sh

RUN chmod +x /app/start.sh

# 포트 노출
EXPOSE 7860

# 볼륨 마운트 포인트 (모델 캐시용)
VOLUME ["/app/pretrained_models"]

# 기본 실행 명령
CMD ["/app/start.sh"]

# K8s에서 probe로 헬스체크를 수행하므로 Dockerfile HEALTHCHECK 제거