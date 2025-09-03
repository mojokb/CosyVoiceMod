# 🐳 CosyVoice Docker 배포 가이드

CosyVoice Gradio 앱을 Docker 컨테이너로 쉽게 배포하고 실행할 수 있습니다.

## 📋 지원 환경

- **GPU 버전**: NVIDIA GPU + CUDA 11.8+ 환경
- **CPU 버전**: CPU 전용 환경 (GPU 없는 서버)
- **플랫폼**: Linux, macOS, Windows (Docker Desktop)

## 🚀 빠른 시작

### 1. 리포지토리 클론

```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
```

### 2. Docker 빌드

```bash
# 자동 빌드 (GPU/CPU 환경 자동 감지)
./docker-build.sh

# 또는 수동 선택
./docker-build.sh 1  # GPU 버전
./docker-build.sh 2  # CPU 버전
./docker-build.sh 3  # 둘 다 빌드
```

### 3. 컨테이너 실행

```bash
# 자동 실행 (GPU/CPU 선택)
./docker-run.sh gpu   # GPU 버전
./docker-run.sh cpu   # CPU 버전

# 백그라운드 실행
./docker-run.sh gpu --detach

# 커스텀 포트
./docker-run.sh gpu --port 8080
```

### 4. 웹 접속

브라우저에서 http://localhost:7860 접속

## 🛠️ 상세 사용법

### Docker Compose 사용

```bash
# GPU 버전 실행
docker-compose up

# CPU 버전 실행
docker-compose --profile cpu up

# 백그라운드 실행
docker-compose up -d

# 종료
docker-compose down
```

### 수동 Docker 실행

#### GPU 버전

```bash
docker run -d \
  --name cosyvoice-gpu \
  --gpus all \
  -p 7860:7860 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  cosyvoice-gradio:latest
```

#### CPU 버전

```bash
docker run -d \
  --name cosyvoice-cpu \
  -p 7860:7860 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  cosyvoice-gradio:cpu
```

## ⚙️ 고급 설정

### 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `GRADIO_SERVER_PORT` | Gradio 서버 포트 | 7860 |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU ID | 0 |
| `DOWNLOAD_LARGE_MODEL` | 대용량 모델 다운로드 | false |
| `OMP_NUM_THREADS` | CPU 스레드 수 | 4 |

### 볼륨 마운트

```bash
docker run -d \
  --name cosyvoice \
  --gpus all \
  -p 7860:7860 \
  -v /path/to/models:/app/pretrained_models \  # 모델 저장소
  -v /path/to/temp:/tmp \                      # 임시 파일
  -v /path/to/cache:/app/.cache \              # 캐시 디렉토리
  cosyvoice-gradio:latest
```

### 포트 설정

```bash
# 커스텀 포트로 실행
docker run -d \
  --name cosyvoice \
  --gpus all \
  -p 8080:7860 \
  -e GRADIO_SERVER_PORT=7860 \
  cosyvoice-gradio:latest
```

## 📊 성능 최적화

### GPU 최적화

```bash
# 특정 GPU 사용
docker run --gpus '"device=0"' ...

# 메모리 제한
docker run --gpus all -m 8g ...

# CPU 제한
docker run --cpus="4.0" ...
```

### CPU 최적화

```bash
# CPU 코어 수 설정
docker run \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  --cpus="8.0" \
  cosyvoice-gradio:cpu
```

## 🔧 문제 해결

### 일반적인 문제

#### 1. GPU 인식 안됨

```bash
# NVIDIA Docker 런타임 확인
docker info | grep nvidia

# GPU 테스트
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

#### 2. 메모리 부족

```bash
# 메모리 사용량 확인
docker stats cosyvoice-gpu

# 메모리 제한 설정
docker run -m 8g --gpus all ...
```

#### 3. 모델 다운로드 실패

```bash
# 컨테이너 로그 확인
docker logs cosyvoice-gpu

# 수동 모델 다운로드
docker exec -it cosyvoice-gpu /app/download_models.sh
```

#### 4. 포트 충돌

```bash
# 포트 사용 확인
netstat -tulpn | grep 7860

# 다른 포트로 실행
./docker-run.sh gpu --port 8080
```

### 디버깅 모드

```bash
# 컨테이너 내부 접속
docker exec -it cosyvoice-gpu bash

# 로그 실시간 확인
docker logs -f cosyvoice-gpu

# 컨테이너 상태 확인
docker inspect cosyvoice-gpu
```

## 🌐 원격 배포

### 클라우드 배포

#### AWS EC2

```bash
# GPU 인스턴스 (p3.2xlarge 권장)
# NVIDIA Driver + Docker + nvidia-docker 설치 후
./docker-run.sh gpu --detach
```

#### Google Cloud Platform

```bash
# GPU VM 인스턴스 생성
gcloud compute instances create cosyvoice-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE

# SSH 접속 후 설치
sudo apt update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# NVIDIA Docker 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 보안 설정

```bash
# HTTPS 프록시 (nginx)
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 지원 (Gradio용)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 자동 재시작

```bash
# systemd 서비스 생성
sudo tee /etc/systemd/system/cosyvoice.service << EOF
[Unit]
Description=CosyVoice Gradio Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/docker-compose -f /path/to/CosyVoice/docker-compose.yml up -d
ExecStop=/usr/local/bin/docker-compose -f /path/to/CosyVoice/docker-compose.yml down
WorkingDirectory=/path/to/CosyVoice

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl enable cosyvoice.service
sudo systemctl start cosyvoice.service
```

## 📁 파일 구조

```
CosyVoice/
├── Dockerfile              # GPU 버전 Dockerfile
├── Dockerfile.cpu          # CPU 버전 Dockerfile  
├── docker-compose.yml      # Docker Compose 설정
├── docker-build.sh         # 빌드 스크립트
├── docker-run.sh           # 실행 스크립트
├── .dockerignore           # Docker 제외 파일
├── gradio_app.py           # Gradio 웹 앱
├── requirements_gradio.txt # Gradio 의존성
└── pretrained_models/      # 모델 저장소 (볼륨 마운트)
    ├── CosyVoice2-0.5B/
    ├── CosyVoice-300M-SFT/
    └── CosyVoice-ttsfrd/
```

## 🔍 모니터링

### 리소스 사용량 확인

```bash
# 컨테이너 리소스 모니터링
docker stats cosyvoice-gpu

# GPU 사용량 확인
docker exec cosyvoice-gpu nvidia-smi

# 로그 분석
docker logs cosyvoice-gpu | grep -i error
```

### 헬스체크

```bash
# 컨테이너 상태 확인
docker ps | grep cosyvoice

# API 응답 테스트
curl -f http://localhost:7860/api/v1/status
```

## 🆘 지원

- **Issues**: [GitHub Issues](https://github.com/FunAudioLLM/CosyVoice/issues)
- **Docker Hub**: (이미지 업로드 시)
- **Documentation**: 이 문서 및 `README_gradio.md`

---

**Happy Dockerizing! 🐳**