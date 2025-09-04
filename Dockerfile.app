# CosyVoice Application Image - RTX 3090 Optimized
# 베이스 이미지를 사용하여 애플리케이션만 설치하는 경량화된 이미지
FROM cosyvoice-base:rtx3090-v2.0

# 애플리케이션 환경 변수
ENV CUDA_VISIBLE_DEVICES=0
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 작업 디렉토리는 이미 베이스에서 설정됨
WORKDIR /app

# CosyVoice 특화 의존성만 설치 (자주 변경되는 것들)
COPY requirements.txt /app/requirements.txt

# CosyVoice 특화 패키지 설치 (베이스에 없는 것들만)
RUN pip install --no-cache-dir \
    # CosyVoice 특화 패키지들
    conformer==0.3.2 \
    HyperPyYAML==1.2.2 \
    lightning==2.2.4 \
    networkx==3.1 \
    pyworld==0.3.4 \
    tensorboard==2.14.0 \
    diffusers==0.29.0 \
    # FastAPI (API 서버용)
    fastapi==0.115.6 \
    fastapi-cli==0.0.4 \
    uvicorn==0.30.0 \
    # 기타 유틸리티
    gdown==5.1.0 \
    pyarrow==18.1.0 \
    pydantic==2.7.0 \
    wetext==0.0.4 \
    wget==3.2

# DeepSpeed는 Linux에서만 설치
RUN if [ "$(uname)" = "Linux" ]; then \
        pip install --no-cache-dir deepspeed==0.15.1; \
    fi

# 애플리케이션 코드 복사
COPY . /app/

# third_party 서브모듈 초기화
RUN if [ -f .gitmodules ]; then \
        git submodule update --init --recursive || echo "서브모듈 초기화 실패 또는 없음"; \
    fi

# Matcha-TTS 경로 생성
RUN mkdir -p third_party/Matcha-TTS

# 모델 다운로드 스크립트 생성 및 실행 권한 설정
COPY <<EOF /app/download_models.sh
#!/bin/bash
set -e

echo "🔄 CosyVoice 모델 다운로드 중..."
mkdir -p /app/pretrained_models

download_model() {
    local model_id="\$1"
    local local_dir="\$2"
    
    if [ ! -d "\$local_dir" ] || [ -z "\$(ls -A "\$local_dir" 2>/dev/null)" ]; then
        echo "📥 \$model_id 다운로드 중..."
        python -c "
from modelscope import snapshot_download
import os
try:
    snapshot_download('\$model_id', local_dir='\$local_dir')
    print('✅ \$model_id 다운로드 완료')
except Exception as e:
    print(f'❌ \$model_id 다운로드 실패: {e}')
    exit(1)
"
    else
        echo "✅ \$model_id 이미 존재"
    fi
}

# 필수 모델들 다운로드
download_model "iic/CosyVoice2-0.5B" "/app/pretrained_models/CosyVoice2-0.5B"
download_model "iic/CosyVoice-300M-SFT" "/app/pretrained_models/CosyVoice-300M-SFT"
download_model "iic/CosyVoice-ttsfrd" "/app/pretrained_models/CosyVoice-ttsfrd"

# 선택적 모델들 (환경변수로 제어)
if [ "\$DOWNLOAD_INSTRUCT_MODEL" = "true" ]; then
    download_model "iic/CosyVoice-300M-Instruct" "/app/pretrained_models/CosyVoice-300M-Instruct"
fi

if [ "\$DOWNLOAD_BASE_MODEL" = "true" ]; then
    download_model "iic/CosyVoice-300M" "/app/pretrained_models/CosyVoice-300M"
fi

echo "🎉 모든 모델 다운로드 완료!"
EOF

RUN chmod +x /app/download_models.sh

# 애플리케이션 시작 스크립트
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

echo "🎙️ CosyVoice RTX 3090 최적화 애플리케이션 시작..."

# GPU 정보 확인
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
    
    # RTX 3090 감지 및 최적화 설정
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -i "rtx.*3090" > /dev/null; then
        echo "🚀 RTX 3090 감지됨 - 최적화 설정 적용"
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
        export OMP_NUM_THREADS=8
        export MKL_NUM_THREADS=8
    fi
else
    echo "⚠️ GPU 미감지 - CPU 모드로 실행"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
fi

# 메모리 및 환경 정보
echo "💾 메모리 정보:"
free -h | head -2

echo "🐍 Python 환경:"
python --version
echo "PyTorch 버전: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 사용 가능: \$(python -c 'import torch; print(torch.cuda.is_available())')"

# 모델 다운로드 확인
echo "📥 모델 확인 및 다운로드..."
/app/download_models.sh

# Gradio 앱 실행
echo "🌐 Gradio 웹 앱 시작..."
echo "🔗 웹 인터페이스: http://localhost:\$GRADIO_SERVER_PORT"
echo "📊 시작 시간: \$(date)"

cd /app
exec python gradio_app.py
EOF

RUN chmod +x /app/start.sh

# 헬스체크 (베이스 이미지에서 curl이 이미 설치됨)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/api/v1/status 2>/dev/null || curl -f http://localhost:7860/ || exit 1

# 포트 노출
EXPOSE 7860

# 볼륨 마운트 포인트 (모델 및 임시 파일용)
VOLUME ["/app/pretrained_models", "/tmp", "/app/.cache"]

# 기본 실행 명령
CMD ["/app/start.sh"]

# 메타데이터
LABEL maintainer="CosyVoice Team"
LABEL version="2.0-app-rtx3090"  
LABEL description="CosyVoice Gradio Application optimized for RTX 3090"
LABEL base.image="cosyvoice-base:rtx3090-v2.0"
LABEL app.type="gradio"
LABEL gpu.architecture="ampere"