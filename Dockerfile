# CosyVoice Gradio Docker Image - RTX 3090 Optimized
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# RTX 3090 최적화 설정
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3090 Ampere architecture

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    # 오디오 처리 라이브러리
    ffmpeg \
    libsndfile1-dev \
    libsox-fmt-all \
    sox \
    # Python 3.10 직접 설치
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 3.10을 기본 Python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# Git LFS 초기화 (모델 다운로드용)
RUN git lfs install

# PyTorch 및 기본 의존성 먼저 설치 (레이어 캐싱 최적화)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# CosyVoice 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app/

# third_party 서브모듈 초기화 (존재하는 경우)
RUN if [ -f .gitmodules ]; then \
        git submodule update --init --recursive || echo "서브모듈 초기화 실패 또는 없음"; \
    fi

# Matcha-TTS 경로 생성
RUN mkdir -p third_party/Matcha-TTS

# 모델 다운로드 스크립트 생성 및 권한 설정
RUN cat > /app/download_models.sh << 'EOF'
#!/bin/bash
set -e

echo "🔄 CosyVoice 모델 다운로드 중..."
mkdir -p /app/pretrained_models

download_model() {
    local model_id="$1"
    local local_dir="$2"
    
    if [ ! -d "$local_dir" ] || [ -z "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "📥 $model_id 다운로드 중..."
        python -c "
from modelscope import snapshot_download
import os
try:
    snapshot_download('$model_id', local_dir='$local_dir')
    print('✅ $model_id 다운로드 완료')
except Exception as e:
    print(f'❌ $model_id 다운로드 실패: {e}')
    exit(1)
"
    else
        echo "✅ $model_id 이미 존재"
    fi
}

# 필수 모델들 다운로드
download_model "iic/CosyVoice2-0.5B" "/app/pretrained_models/CosyVoice2-0.5B"
download_model "iic/CosyVoice-300M-SFT" "/app/pretrained_models/CosyVoice-300M-SFT"
download_model "iic/CosyVoice-ttsfrd" "/app/pretrained_models/CosyVoice-ttsfrd"

# 선택적 모델들 (환경변수로 제어)
if [ "$DOWNLOAD_INSTRUCT_MODEL" = "true" ]; then
    download_model "iic/CosyVoice-300M-Instruct" "/app/pretrained_models/CosyVoice-300M-Instruct"
fi

if [ "$DOWNLOAD_BASE_MODEL" = "true" ]; then
    download_model "iic/CosyVoice-300M" "/app/pretrained_models/CosyVoice-300M"
fi

echo "🎉 모든 모델 다운로드 완료!"
EOF

RUN chmod +x /app/download_models.sh

# 컨테이너 시작 스크립트 생성
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "🎙️ CosyVoice RTX 3090 최적화 컨테이너 시작..."

# GPU 정보 확인
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
    
    # RTX 3090 감지
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -i "rtx.*3090" > /dev/null; then
        echo "🚀 RTX 3090 감지됨 - 최적화 설정 적용"
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
    fi
else
    echo "⚠️ GPU 미감지 - CPU 모드로 실행"
fi

# 메모리 정보
echo "💾 메모리 정보:"
free -h

# Python 환경 확인
echo "🐍 Python 환경:"
python --version
pip list | grep -E "(torch|gradio|whisper)"

# 모델 다운로드
echo "📥 모델 확인 및 다운로드..."
/app/download_models.sh

# Gradio 앱 실행
echo "🌐 Gradio 웹 앱 시작..."
echo "🔗 웹 인터페이스: http://localhost:7860"

cd /app
python gradio_app.py
EOF

RUN chmod +x /app/start.sh

# 헬스체크 스크립트
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 포트 노출
EXPOSE 7860

# 볼륨 마운트 포인트
VOLUME ["/app/pretrained_models", "/tmp"]

# 기본 실행 명령
CMD ["/app/start.sh"]

# 메타데이터
LABEL maintainer="CosyVoice Team"
LABEL version="2.0-rtx3090"
LABEL description="CosyVoice Gradio App optimized for RTX 3090"
LABEL gpu.architecture="ampere"
LABEL gpu.compute="8.6"