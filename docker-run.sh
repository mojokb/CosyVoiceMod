#!/bin/bash

# CosyVoice Docker 실행 스크립트

set -e

echo "🐳 CosyVoice Docker 실행 스크립트"
echo "================================="

# 사용법 출력
show_usage() {
    echo ""
    echo "사용법: $0 [gpu|cpu] [options]"
    echo ""
    echo "모드:"
    echo "  gpu     GPU 가속 버전 실행"
    echo "  cpu     CPU 전용 버전 실행"
    echo ""
    echo "옵션:"
    echo "  --port PORT     포트 번호 (기본값: 7860)"
    echo "  --models DIR    모델 디렉토리 경로 (기본값: ./pretrained_models)"
    echo "  --detach        백그라운드 실행"
    echo "  --build         이미지가 없으면 자동 빌드"
    echo ""
    echo "예시:"
    echo "  $0 gpu                    # GPU 버전 실행"
    echo "  $0 cpu --port 8080        # CPU 버전을 8080 포트로 실행"
    echo "  $0 gpu --detach           # GPU 버전 백그라운드 실행"
    exit 1
}

# 기본값 설정
MODE=""
PORT="7860"
MODELS_DIR="$(pwd)/pretrained_models"
DETACH=""
AUTO_BUILD=false

# 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        gpu|cpu)
            MODE="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --models)
            MODELS_DIR="$2"
            shift 2
            ;;
        --detach)
            DETACH="-d"
            shift
            ;;
        --build)
            AUTO_BUILD=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "❌ 알 수 없는 옵션: $1"
            show_usage
            ;;
    esac
done

# 모드 선택
if [[ -z "$MODE" ]]; then
    echo ""
    echo "실행 모드를 선택하세요:"
    echo "1) GPU 가속 버전 (CUDA 필요)"
    echo "2) CPU 전용 버전"
    echo ""
    read -p "선택 (1-2): " choice
    
    case $choice in
        1) MODE="gpu" ;;
        2) MODE="cpu" ;;
        *) 
            echo "❌ 잘못된 선택입니다."
            exit 1
            ;;
    esac
fi

# 이미지 이름 설정
if [[ "$MODE" == "gpu" ]]; then
    IMAGE_NAME="cosyvoice-gradio:latest"
    CONTAINER_NAME="cosyvoice-gpu"
    RUNTIME_ARGS="--gpus all"
else
    IMAGE_NAME="cosyvoice-gradio:cpu"
    CONTAINER_NAME="cosyvoice-cpu"
    RUNTIME_ARGS=""
fi

# 이미지 존재 확인
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "❌ Docker 이미지 '$IMAGE_NAME'를 찾을 수 없습니다."
    
    if [[ "$AUTO_BUILD" == "true" ]]; then
        echo "🔨 자동 빌드를 시작합니다..."
        if [[ "$MODE" == "gpu" ]]; then
            ./docker-build.sh 1
        else
            ./docker-build.sh 2
        fi
    else
        echo ""
        echo "다음 명령어로 이미지를 빌드하세요:"
        echo "  ./docker-build.sh"
        echo ""
        echo "또는 --build 옵션을 사용하여 자동 빌드:"
        echo "  $0 $MODE --build"
        exit 1
    fi
fi

# 모델 디렉토리 생성
mkdir -p "$MODELS_DIR"

# 기존 컨테이너 정리
if docker ps -a --format 'table {{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "🧹 기존 컨테이너 '$CONTAINER_NAME' 정리 중..."
    docker rm -f "$CONTAINER_NAME" > /dev/null
fi

echo ""
echo "🚀 CosyVoice 컨테이너 시작 중..."
echo "모드: $MODE"
echo "포트: $PORT"
echo "모델 디렉토리: $MODELS_DIR"

# Docker 실행
docker run $DETACH \
    --name "$CONTAINER_NAME" \
    $RUNTIME_ARGS \
    -p "$PORT:7860" \
    -v "$MODELS_DIR:/app/pretrained_models" \
    -v "/tmp/gradio:/tmp" \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    -e GRADIO_SERVER_PORT=7860 \
    --restart unless-stopped \
    "$IMAGE_NAME"

if [[ -z "$DETACH" ]]; then
    echo ""
    echo "🎉 CosyVoice가 시작되었습니다!"
    echo "🌐 웹 인터페이스: http://localhost:$PORT"
    echo ""
    echo "⏹️  종료하려면 Ctrl+C를 누르세요"
else
    echo ""
    echo "🎉 CosyVoice가 백그라운드에서 시작되었습니다!"
    echo "🌐 웹 인터페이스: http://localhost:$PORT"
    echo ""
    echo "📋 컨테이너 상태 확인:"
    echo "  docker ps | grep $CONTAINER_NAME"
    echo ""
    echo "📋 로그 확인:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    echo "⏹️  컨테이너 중지:"
    echo "  docker stop $CONTAINER_NAME"
fi