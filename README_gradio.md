# 🎙️ CosyVoice Gradio 웹 앱

브라우저에서 음성 녹음 → STT → TTS를 한 번에 수행할 수 있는 웹 인터페이스입니다.

## ✨ 주요 기능

- 🎤 **브라우저 마이크 녹음**: 웹에서 바로 음성 녹음
- 🎯 **자동 STT**: Whisper로 음성을 텍스트로 자동 변환
- 🎵 **음성 복제**: 사용자 목소리로 새로운 텍스트 음성화
- 🌍 **다국어 지원**: 한국어, 영어, 중국어, 일본어
- 🚀 **원클릭 생성**: 자동 모드로 한 번에 처리

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# CosyVoice 환경 활성화
conda activate cosyvoice

# Gradio 추가 의존성 설치
pip install -r requirements_gradio.txt
```

### 2. 모델 다운로드 및 실행

```bash
# 자동 실행 (모델 다운로드 포함)
./run_gradio.sh

# 또는 수동 실행
python gradio_app.py
```

### 3. 웹 접속

브라우저에서 http://localhost:7860 으로 접속

## 📱 사용법

### 🚀 자동 생성 모드 (추천)

1. **참조 음성 녹음** 📢
   - 마이크 버튼 클릭
   - 5-10초간 명확하게 발음
   - 노이즈가 적은 환경에서 녹음

2. **텍스트 입력** 📝
   - 생성하고 싶은 텍스트 입력
   - 한국어/영어/중국어/일본어 지원

3. **모델 선택** 🤖
   - CosyVoice2-0.5B (추천) - 최고 성능
   - CosyVoice-300M-SFT - 빠른 처리

4. **생성** 🎵
   - "음성 생성하기" 버튼 클릭
   - 결과 음성 재생 및 다운로드

### 🔧 수동 설정 모드

고급 사용자를 위한 단계별 제어:

1. **STT 단계**: 음성 인식 결과 확인/수정
2. **TTS 단계**: 세밀한 설정으로 음성 생성

## 🔧 고급 설정

### 모델별 특징

| 모델 | 특징 | 권장 용도 |
|------|------|-----------|
| **CosyVoice2-0.5B** | 최신, 최고 품질 | 일반적인 음성 복제 |
| **CosyVoice-300M-SFT** | 빠름, 미리 학습된 화자 | 빠른 테스트 |
| **CosyVoice-300M-Instruct** | 감정/스타일 제어 | 고급 표현 |

### Whisper 모델 크기

| 크기 | 속도 | 정확도 | 메모리 |
|------|------|--------|--------|
| **tiny** | 매우 빠름 | 낮음 | 적음 |
| **base** | 빠름 | 좋음 | 적당 |
| **small** | 보통 | 매우 좋음 | 보통 |
| **medium** | 느림 | 최고 | 많음 |

## 🎯 최적 사용 팁

### 📢 좋은 참조 음성 녹음
```
✅ 권장:
- 5-10초 길이
- 명확한 발음
- 조용한 환경
- 자연스러운 말투

❌ 피할 것:
- 너무 짧은 음성 (<3초)
- 배경 노이즈가 많은 환경
- 중얼거리는 발음
- 음악이나 효과음이 섞인 음성
```

### 📝 텍스트 입력 팁
```
✅ 권장:
- 자연스러운 문장
- 적절한 길이 (1-3문장)
- 표준어 사용

❌ 피할 것:
- 너무 긴 텍스트 (>200자)
- 특수 기호 남발
- 줄임말/은어 과다
```

## 🔧 문제 해결

### 일반적인 오류

1. **"CosyVoice가 설치되지 않았습니다"**
   ```bash
   # CosyVoice 설치 확인
   pip install -r requirements.txt
   ```

2. **"모델을 찾을 수 없습니다"**
   ```bash
   # 모델 다운로드
   ./run_gradio.sh
   ```

3. **"CUDA 메모리 부족"**
   - 더 작은 모델 사용 (CosyVoice-300M)
   - 다른 GPU 프로세스 종료

4. **"마이크 접근 권한 없음"**
   - 브라우저 마이크 권한 허용
   - HTTPS 환경에서 실행

### 성능 최적화

```bash
# GPU 가속 (CUDA 사용 가능시)
export CUDA_VISIBLE_DEVICES=0

# CPU 코어 수 설정
export OMP_NUM_THREADS=4

# 메모리 사용량 제한
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🌐 원격 접속

### 로컬 네트워크에서 접속
```python
# gradio_app.py 마지막 부분 수정
demo.launch(
    server_name="0.0.0.0",  # 모든 IP에서 접속 허용
    server_port=7860,
    share=False  # 보안상 False 권장
)
```

### 공개 URL 생성 (임시)
```python
demo.launch(
    share=True  # Gradio 공개 URL 생성
)
```

## 📁 파일 구조

```
CosyVoice/
├── gradio_app.py           # 메인 Gradio 앱
├── run_gradio.sh          # 자동 실행 스크립트
├── requirements_gradio.txt # Gradio 전용 의존성
├── README_gradio.md       # 이 파일
└── pretrained_models/     # 모델 파일들
    ├── CosyVoice2-0.5B/
    ├── CosyVoice-300M-SFT/
    └── CosyVoice-ttsfrd/
```

## 🆘 지원

- **Issues**: [GitHub Issues](https://github.com/FunAudioLLM/CosyVoice/issues)
- **Documentation**: [CosyVoice 공식 문서](https://github.com/FunAudioLLM/CosyVoice)
- **Tutorial**: `tutorial.md` 참고

---

**즐거운 음성 생성 되세요! 🎵**