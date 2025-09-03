#!/bin/bash

# CosyVoice Gradio 앱 실행 스크립트

echo "🎙️ CosyVoice Gradio 앱을 시작합니다..."

# Python 가상환경 활성화 (필요시)
# source venv/bin/activate

# 모델 다운로드 확인
if [ ! -d "pretrained_models" ]; then
    echo "📥 모델을 다운로드합니다..."
    mkdir -p pretrained_models
    
    echo "CosyVoice2-0.5B 다운로드 중..."
    python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
    
    echo "CosyVoice-300M-SFT 다운로드 중..."
    python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')"
    
    echo "CosyVoice-ttsfrd 다운로드 중..."
    python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')"
fi

# Gradio 앱 실행
echo "🚀 웹 앱을 시작합니다..."
python gradio_app.py