# 🐳 CosyVoice Docker 설정 가이드

CosyVoice를 Docker를 통해 실행하는 방법을 안내합니다. 베이스 이미지와 애플리케이션 이미지로 분리하여 효율적인 빌드와 배포가 가능합니다.

## 📁 Docker 구조

```
├── Dockerfile.base     # CosyVoice 패키지와 의존성을 포함한 베이스 이미지
├── Dockerfile.app      # Gradio 애플리케이션만을 포함한 경량 이미지
├── build.sh           # 자동 빌드 스크립트
├── docker-compose.yml # 도커 컴포즈 설정
└── README-Docker.md   # 이 파일
```

## 🏗️ 베이스/앱 분리 구조의 장점

### Dockerfile.base
- **포함 내용**: PyTorch, CosyVoice 패키지, 시스템 의존성
- **특징**: 큰 용량, 변경 빈도 낮음
- **용도**: 개발 환경 전체를 포함한 안정적 베이스

### Dockerfile.app  
- **포함 내용**: Gradio 웹 애플리케이션 코드만
- **특징**: 작은 용량, 변경 빈도 높음
- **용도**: 빠른 배포와 업데이트

## 🚀 빠른 시작

### 1. 자동 빌드 스크립트 사용 (추천)

```bash
# 전체 빌드 (베이스 + 앱)
./build.sh -f

# 베이스만 빌드
./build.sh -b

# 앱만 빌드 (베이스가 이미 있는 경우)
./build.sh -a

# 캐시 없이 전체 빌드
./build.sh -f --no-cache
```

### 2. Docker Compose 사용

```bash
# GPU 버전 실행
docker-compose up

# CPU 버전 실행  
docker-compose --profile cpu up

# 백그라운드 실행
docker-compose up -d

# Nginx 프록시와 함께 실행
docker-compose --profile nginx up
```

### 3. 수동 Docker 명령

```bash
# 1. 베이스 이미지 빌드
docker build -f Dockerfile.base -t cosyvoice-base:rtx3090-v2.0 .

# 2. 앱 이미지 빌드
docker build -f Dockerfile.app -t cosyvoice-app:rtx3090-v2.0 .

# 3. 실행
docker run -p 7860:7860 --gpus all cosyvoice-app:rtx3090-v2.0
```

## 🎛️ 빌드 스크립트 옵션

`build.sh` 스크립트는 다양한 옵션을 제공합니다:

```bash
Usage: ./build.sh [OPTIONS]

Options:
  -b, --base-only       베이스 이미지만 빌드
  -a, --app-only        애플리케이션 이미지만 빌드  
  -f, --full            전체 빌드 (베이스 + 앱)
  --no-cache            캐시 없이 빌드
  --push                빌드 후 레지스트리에 푸시
  --tag-suffix SUFFIX   이미지 태그에 접미사 추가
  -h, --help            도움말 표시

Examples:
  ./build.sh -f                  # 전체 빌드
  ./build.sh -b --no-cache       # 베이스 이미지를 캐시 없이 빌드
  ./build.sh -a --push           # 앱 이미지 빌드 후 푸시
  ./build.sh -f --tag-suffix dev # 개발용 태그로 전체 빌드
```

## 🔧 환경 변수 설정

### GPU 최적화 설정
```bash
CUDA_VISIBLE_DEVICES=0                    # 사용할 GPU 지정
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024  # 메모리 최적화
OMP_NUM_THREADS=8                         # CPU 스레드 수
MKL_NUM_THREADS=8                         # MKL 스레드 수
```

### 모델 다운로드 제어
```bash
DOWNLOAD_INSTRUCT_MODEL=true   # Instruct 모델 다운로드
DOWNLOAD_BASE_MODEL=true       # Base 모델 다운로드  
```

### Gradio 설정
```bash
GRADIO_SERVER_NAME=0.0.0.0     # 서버 바인딩 주소
GRADIO_SERVER_PORT=7860        # 서버 포트
GRADIO_SHARE=false             # 공개 링크 생성 여부
GRADIO_INBROWSER=false         # 자동 브라우저 열기 여부
```

## 🚀 성능 최적화 (NEW)

### 모델 다운로드 최적화

**기존 문제점:**
- 컨테이너 시작 시마다 모델 다운로드 확인/실행
- 네트워크 의존적 런타임 로딩
- 긴 컨테이너 시작 시간

**최적화된 구조:**
```
cosyvoice-base:rtx3090-v2.0    ←── 필수 모델 사전 다운로드
    └── cosyvoice-app:latest   ←── 경량화된 앱만 포함
```

### 최적화 결과

| 항목 | 기존 | 최적화 후 |
|------|------|-----------|
| 첫 빌드 | ~30분 | 베이스: ~30분, 앱: ~2분 |
| 재빌드 | ~30분 | ~2분 (앱만) |
| 컨테이너 시작 | 2-5분 | 10-30초 |
| 네트워크 의존성 | 높음 | 없음 |

## 📂 볼륨 마운트 전략

### 권장 구조 (외부 모델 관리)
```bash
mkdir -p ./models
docker run -p 7860:7860 --gpus all \
  -v ./models:/app/pretrained_models \
  cosyvoice-app:latest
```

### 볼륨 마운트 장점
- **영구 저장**: 컨테이너 재생성해도 모델 보존
- **공유 가능**: 여러 컨테이너가 같은 모델 사용  
- **업데이트 용이**: 호스트에서 모델 관리 가능
- **빠른 시작**: 다운로드 없이 즉시 실행

| 호스트 경로 | 컨테이너 경로 | 설명 |
|-------------|---------------|------|
| `./models` | `/app/pretrained_models` | 모델 파일 저장 (권장) |
| `./cache` | `/app/.cache` | 패키지 캐시 |
| `/tmp` | `/tmp` | 임시 파일 |
| `./logs` | `/app/logs` | 로그 파일 (선택적) |

## 🌐 포트 설정

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Gradio Web UI | 7860 | 메인 웹 인터페이스 |
| FastAPI | 8000 | API 서버 (선택적) |
| CPU 버전 | 7861 | CPU 전용 버전 |
| Nginx | 80/443 | 리버스 프록시 |

## 🏥 헬스체크

컨테이너 상태를 모니터링하는 헬스체크가 설정되어 있습니다:

- **확인 URL**: `http://localhost:7860/`
- **확인 주기**: 30초
- **타임아웃**: 10초
- **재시도**: 3회
- **시작 대기**: 90초 (GPU), 180초 (CPU)

## 🔍 사용법 예제

### 개발 환경 설정
```bash
# 개발용 태그로 빌드
./build.sh -f --tag-suffix dev

# 개발용 컨테이너 실행
docker run -p 7860:7860 --gpus all \
  -v $(pwd):/app \
  cosyvoice-app:rtx3090-v2.0-dev
```

### 운영 환경 배포
```bash
# 운영용 빌드 및 푸시
./build.sh -f --tag-suffix prod --push

# 운영 환경에서 실행
docker-compose -f docker-compose.prod.yml up -d
```

### CPU 전용 환경
```bash
# CPU 프로필로 실행
docker-compose --profile cpu up

# 또는 직접 실행
docker run -p 7861:7860 \
  -e CUDA_VISIBLE_DEVICES="" \
  cosyvoice-app:cpu
```

## 🐛 트러블슈팅

### 자주 발생하는 문제들

#### 1. GPU 메모리 부족
```bash
# 환경변수로 메모리 제한 설정
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

#### 2. 모델 다운로드 실패
```bash
# 수동으로 모델 다운로드
mkdir -p pretrained_models
docker run --rm -v $(pwd)/pretrained_models:/models \
  python:3.10 python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='/models/CosyVoice2-0.5B')
"
```

#### 3. 권한 문제
```bash
# 볼륨 디렉토리 권한 수정
sudo chown -R $USER:$USER ./pretrained_models ./cache ./logs
```

#### 4. 포트 충돌
```bash
# 다른 포트로 실행
docker run -p 8080:7860 --gpus all cosyvoice-app:rtx3090-v2.0
```

### 로그 확인
```bash
# 컨테이너 로그 확인
docker-compose logs -f cosyvoice

# 특정 서비스 로그만 확인
docker-compose logs -f cosyvoice --tail=100
```

## 🔧 고급 설정

### 멀티 GPU 설정
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # 2개 GPU 사용
          capabilities: [gpu]
```

### 리소스 제한
```yaml
mem_limit: 16g        # 메모리 제한
memswap_limit: 16g    # 스왑 제한  
shm_size: 2g          # 공유 메모리 크기
```

### 네트워크 설정
```yaml
networks:
  cosyvoice-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 📊 성능 모니터링

### 리소스 사용량 확인
```bash
# 실시간 모니터링
docker stats cosyvoice-gradio

# GPU 사용량 확인 (컨테이너 내부)
docker exec cosyvoice-gradio nvidia-smi
```

### 성능 최적화 팁

1. **SSD 사용**: 모델 파일을 SSD에 저장
2. **충분한 RAM**: 최소 16GB 권장
3. **GPU 메모리**: RTX 3090 (24GB) 권장
4. **네트워크**: 빠른 인터넷 (모델 다운로드용)

이 가이드를 통해 CosyVoice를 효율적으로 Docker 환경에서 실행할 수 있습니다. 추가 질문이나 문제가 있으면 GitHub Issues를 참고하세요.