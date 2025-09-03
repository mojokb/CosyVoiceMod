# CosyVoice ComfyUI Custom Node 개발 가이드

## 1. 프로젝트 개요

CosyVoice는 Alibaba에서 개발한 Text-to-Speech(TTS) 모델로, 다국어 지원과 zero-shot 음성 합성 기능을 제공합니다.

### 1.1 핵심 특징
- **다국어 지원**: 중국어, 영어, 일본어, 한국어, 중국 방언
- **Zero-shot 음성 복제**: 짧은 프롬프트 음성으로 화자 특성 학습
- **스트리밍 지원**: 실시간 음성 생성
- **감정 제어**: 웃음, 호흡 등 세밀한 감정 표현
- **다중 모델**: CosyVoice 1.0, 2.0, 3.0 버전 지원

## 2. 핵심 아키텍처 분석

### 2.1 주요 컴포넌트

#### CosyVoiceFrontEnd (`cosyvoice/cli/frontend.py`)
- **기능**: 텍스트 전처리 및 음성 토큰 추출
- **핵심 메서드**:
  - `text_normalize()`: 텍스트 정규화 
  - `_extract_text_token()`: 텍스트를 토큰으로 변환
  - `_extract_speech_token()`: 음성을 토큰으로 변환
  - `_extract_spk_embedding()`: 화자 임베딩 추출

#### CosyVoice/CosyVoice2 (`cosyvoice/cli/cosyvoice.py`)
- **기능**: 메인 TTS 인터페이스
- **핵심 메서드**:
  - `inference_sft()`: SFT 모델 추론
  - `inference_zero_shot()`: Zero-shot 음성 복제
  - `inference_cross_lingual()`: 다국어 음성 합성
  - `inference_instruct()`: 지시 기반 음성 생성

#### TransformerLM (`cosyvoice/llm/llm.py`)  
- **기능**: 언어 모델 기반 음성 토큰 생성
- **구성**: Text Encoder + LLM + 음성 토큰 디코더

#### MaskedDiffWithXvec (`cosyvoice/flow/flow.py`)
- **기능**: Flow-based 음성 특징 생성
- **구성**: Encoder + Length Regulator + Decoder

#### HiFiGan (`cosyvoice/hifigan/hifigan.py`)
- **기능**: 멜-스펙트로그램에서 음성 파형 생성
- **구성**: Generator + Discriminator

### 2.2 데이터 플로우

```
텍스트 → 전처리 → 텍스트 토큰 → LLM → 음성 토큰 → Flow → 멜 스펙트로그램 → HiFiGan → 음성 파형
```

## 3. ComfyUI 노드 구현 가이드

### 3.1 기본 노드 구조

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "CosyVoice"))
sys.path.append(os.path.join(os.path.dirname(__file__), "CosyVoice", "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import numpy as np

class CosyVoiceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "pretrained_models/CosyVoice2-0.5B"}),
                "text": ("STRING", {"default": "안녕하세요, CosyVoice입니다."}),
                "mode": (["sft", "zero_shot", "cross_lingual", "instruct"], {"default": "sft"}),
            },
            "optional": {
                "speaker_id": ("STRING", {"default": "중문여"}),
                "prompt_text": ("STRING", {"default": ""}),
                "prompt_audio": ("AUDIO",),
                "instruct": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/tts"
    
    def __init__(self):
        self.model = None
        self.current_model_path = None
    
    def load_model(self, model_path):
        if self.current_model_path != model_path:
            if "CosyVoice2" in model_path:
                self.model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
            else:
                self.model = CosyVoice(model_path, load_jit=False, load_trt=False, fp16=False)
            self.current_model_path = model_path
    
    def generate_speech(self, model_path, text, mode, speaker_id="", prompt_text="", 
                       prompt_audio=None, instruct=""):
        self.load_model(model_path)
        
        output_audio = None
        
        if mode == "sft":
            for i, j in enumerate(self.model.inference_sft(text, speaker_id, stream=False)):
                output_audio = j['tts_speech'].cpu().numpy()
                break
        
        elif mode == "zero_shot" and prompt_audio is not None:
            prompt_speech_16k = self.prepare_prompt_audio(prompt_audio)
            for i, j in enumerate(self.model.inference_zero_shot(text, prompt_text, 
                                                               prompt_speech_16k, stream=False)):
                output_audio = j['tts_speech'].cpu().numpy()
                break
        
        elif mode == "cross_lingual" and prompt_audio is not None:
            prompt_speech_16k = self.prepare_prompt_audio(prompt_audio)
            for i, j in enumerate(self.model.inference_cross_lingual(text, 
                                                                   prompt_speech_16k, stream=False)):
                output_audio = j['tts_speech'].cpu().numpy()
                break
        
        elif mode == "instruct" and hasattr(self.model, 'inference_instruct2'):
            prompt_speech_16k = self.prepare_prompt_audio(prompt_audio) if prompt_audio else None
            for i, j in enumerate(self.model.inference_instruct2(text, instruct, 
                                                               prompt_speech_16k, stream=False)):
                output_audio = j['tts_speech'].cpu().numpy()
                break
        
        if output_audio is not None:
            # ComfyUI AUDIO 형식으로 변환
            audio_dict = {
                "waveform": torch.from_numpy(output_audio).unsqueeze(0),
                "sample_rate": self.model.sample_rate
            }
            return (audio_dict,)
        
        return (None,)
    
    def prepare_prompt_audio(self, prompt_audio):
        """ComfyUI AUDIO 형식을 CosyVoice 형식으로 변환"""
        if isinstance(prompt_audio, dict):
            waveform = prompt_audio["waveform"]
            sample_rate = prompt_audio["sample_rate"]
            
            # 16kHz로 리샘플링
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            return waveform.squeeze(0).numpy()
        
        return prompt_audio
```

### 3.2 고급 노드 기능

#### 3.2.1 스피커 관리 노드

```python
class CosyVoiceSpeakerManager:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "action": (["list", "add", "save"], {"default": "list"}),
            },
            "optional": {
                "speaker_name": ("STRING", {"default": "custom_speaker"}),
                "prompt_text": ("STRING", {"default": ""}),
                "prompt_audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("STRING", "COSYVOICE_MODEL")
    FUNCTION = "manage_speakers"
    
    def manage_speakers(self, model, action, speaker_name="", prompt_text="", prompt_audio=None):
        if action == "list":
            speakers = model.list_available_spks()
            return ("\n".join(speakers), model)
        
        elif action == "add" and prompt_audio is not None:
            prompt_speech_16k = self.prepare_prompt_audio(prompt_audio)
            success = model.add_zero_shot_spk(prompt_text, prompt_speech_16k, speaker_name)
            return (f"Speaker {speaker_name} added: {success}", model)
        
        elif action == "save":
            model.save_spkinfo()
            return ("Speaker info saved", model)
        
        return ("No action performed", model)
```

#### 3.2.2 배치 처리 노드

```python
class CosyVoiceBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "texts": ("STRING", {"multiline": True}),
                "mode": (["sft", "zero_shot"], {"default": "sft"}),
                "speaker_id": ("STRING", {"default": "중문여"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("batch_audio",)
    FUNCTION = "generate_batch"
    
    def generate_batch(self, model, texts, mode, speaker_id):
        text_list = [t.strip() for t in texts.split('\n') if t.strip()]
        audio_outputs = []
        
        for text in text_list:
            if mode == "sft":
                for i, j in enumerate(model.inference_sft(text, speaker_id, stream=False)):
                    audio_outputs.append(j['tts_speech'].cpu().numpy())
                    break
        
        # 오디오 연결
        if audio_outputs:
            combined_audio = np.concatenate(audio_outputs)
            audio_dict = {
                "waveform": torch.from_numpy(combined_audio).unsqueeze(0),
                "sample_rate": model.sample_rate
            }
            return (audio_dict,)
        
        return (None,)
```

### 3.3 노드 등록

```python
# __init__.py
NODE_CLASS_MAPPINGS = {
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceSpeakerManager": CosyVoiceSpeakerManager,
    "CosyVoiceBatch": CosyVoiceBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoiceNode": "CosyVoice TTS",
    "CosyVoiceSpeakerManager": "CosyVoice Speaker Manager", 
    "CosyVoiceBatch": "CosyVoice Batch TTS",
}
```

## 4. 설치 및 배포

### 4.1 디렉토리 구조

```
ComfyUI/custom_nodes/ComfyUI-CosyVoice/
├── __init__.py
├── cosyvoice_nodes.py
├── requirements.txt
├── CosyVoice/  # Git submodule 또는 복사
└── pretrained_models/
    ├── CosyVoice2-0.5B/
    ├── CosyVoice-300M-SFT/
    └── CosyVoice-ttsfrd/
```

### 4.2 requirements.txt

```
torch>=2.0.0
torchaudio
transformers>=4.21.0
onnxruntime
whisper-openai
hyperpyyaml
modelscope
tqdm
inflect
omegaconf
```

### 4.3 자동 모델 다운로드

```python
def download_models():
    from modelscope import snapshot_download
    
    models = [
        'iic/CosyVoice2-0.5B',
        'iic/CosyVoice-300M-SFT',
        'iic/CosyVoice-ttsfrd'
    ]
    
    for model in models:
        model_dir = f'pretrained_models/{model.split("/")[1]}'
        if not os.path.exists(model_dir):
            snapshot_download(model, local_dir=model_dir)
```

## 5. 성능 최적화

### 5.1 GPU 가속화
- JIT 컴파일 활용: `load_jit=True`
- TensorRT 최적화: `load_trt=True` 
- FP16 추론: `fp16=True`

### 5.2 메모리 관리
- 모델 공유: 인스턴스 레벨에서 모델 캐싱
- 스트리밍 모드: 긴 텍스트에서 메모리 절약
- 배치 크기 제한: OOM 방지

### 5.3 속도 최적화
- 모델 예열: 첫 번째 추론 전 dummy 실행
- 캐시 활용: 중복 토큰화 방지
- 병렬 처리: 배치 모드에서 멀티프로세싱

## 6. 고려사항

### 6.1 호환성
- PyTorch 버전 호환성 확인
- CUDA 버전 매칭
- ComfyUI 버전별 AUDIO 형식 차이

### 6.2 에러 처리
- 모델 로딩 실패 시 fallback
- 메모리 부족 시 자동 복구
- 잘못된 입력에 대한 검증

### 6.3 사용자 경험
- 진행 상황 표시
- 에러 메시지 다국어화
- 모델별 지원 기능 안내

이 가이드를 바탕으로 CosyVoice의 강력한 TTS 기능을 ComfyUI에서 활용할 수 있는 커스텀 노드를 개발할 수 있습니다.