#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CosyVoice Gradio Web App
브라우저에서 음성 녹음 → STT → TTS 생성을 위한 웹 인터페이스
"""

import sys
import os
import gradio as gr
import torch
import torchaudio
import whisper
import numpy as np
import tempfile
from pathlib import Path
import logging

# CosyVoice 경로 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "third_party" / "Matcha-TTS"))

try:
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    COSYVOICE_AVAILABLE = True
except ImportError as e:
    print(f"CosyVoice import error: {e}")
    COSYVOICE_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CosyVoiceGradioApp:
    def __init__(self):
        self.whisper_model = None
        self.cosyvoice_model = None
        self.current_model_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_whisper(self, model_size="base"):
        """Whisper 모델 로드"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
        return self.whisper_model
    
    def load_cosyvoice(self, model_path):
        """CosyVoice 모델 로드"""
        if not COSYVOICE_AVAILABLE:
            raise Exception("CosyVoice가 설치되지 않았습니다.")
        
        if self.current_model_path != model_path:
            logger.info(f"Loading CosyVoice model: {model_path}")
            try:
                if "CosyVoice2" in model_path:
                    self.cosyvoice_model = CosyVoice2(model_path, 
                                                     load_jit=False, 
                                                     load_trt=False, 
                                                     fp16=False)
                else:
                    self.cosyvoice_model = CosyVoice(model_path, 
                                                    load_jit=False, 
                                                    load_trt=False, 
                                                    fp16=False)
                self.current_model_path = model_path
                logger.info("CosyVoice model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CosyVoice: {e}")
                raise
        
        return self.cosyvoice_model
    
    def transcribe_audio(self, audio_file, whisper_size="base", language="auto"):
        """음성을 텍스트로 변환"""
        try:
            if audio_file is None:
                return "❌ 음성 파일을 먼저 녹음해주세요."
            
            # Whisper 모델 로드
            model = self.load_whisper(whisper_size)
            
            # 오디오 파일 처리
            if isinstance(audio_file, str):
                # 파일 경로인 경우
                audio_path = audio_file
            else:
                # 튜플인 경우 (sample_rate, audio_data)
                sample_rate, audio_data = audio_file
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    # 정규화
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    
                    # 16kHz로 리샘플링
                    if sample_rate != 16000:
                        import librosa
                        audio_data = librosa.resample(audio_data, 
                                                    orig_sr=sample_rate, 
                                                    target_sr=16000)
                    
                    torchaudio.save(tmp_file.name, 
                                  torch.from_numpy(audio_data).unsqueeze(0), 
                                  16000)
                    audio_path = tmp_file.name
            
            # STT 실행
            if language == "auto":
                result = model.transcribe(audio_path)
            else:
                result = model.transcribe(audio_path, language=language)
            
            transcribed_text = result["text"].strip()
            
            # 임시 파일 정리
            if 'tmp_file' in locals():
                os.unlink(audio_path)
            
            logger.info(f"Transcription result: {transcribed_text}")
            return f"✅ 인식된 텍스트: {transcribed_text}"
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"❌ 음성 인식 오류: {str(e)}"
    
    def generate_speech(self, reference_audio, prompt_text, target_text, 
                       model_path, mode="zero_shot", speaker_id="중문여"):
        """음성 생성"""
        try:
            if not target_text.strip():
                return None, "❌ 생성할 텍스트를 입력해주세요."
            
            # CosyVoice 모델 로드
            model = self.load_cosyvoice(model_path)
            
            if mode == "sft":
                # SFT 모드 - 미리 학습된 화자 사용
                output_audio = None
                for i, result in enumerate(model.inference_sft(target_text, speaker_id, stream=False)):
                    output_audio = result['tts_speech'].cpu().numpy()
                    break
                
                if output_audio is not None:
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        torchaudio.save(tmp_file.name, 
                                      torch.from_numpy(output_audio), 
                                      model.sample_rate)
                        return tmp_file.name, "✅ 음성이 생성되었습니다 (SFT 모드)"
            
            elif mode == "zero_shot":
                # Zero-shot 모드 - 참조 음성 필요
                if reference_audio is None:
                    return None, "❌ Zero-shot 모드에서는 참조 음성이 필요합니다."
                
                if not prompt_text.strip():
                    return None, "❌ 참조 음성의 텍스트를 입력해주세요."
                
                # 참조 음성 처리
                reference_speech = self.prepare_reference_audio(reference_audio)
                
                output_audio = None
                for i, result in enumerate(model.inference_zero_shot(
                    target_text, prompt_text, reference_speech, stream=False)):
                    output_audio = result['tts_speech'].cpu().numpy()
                    break
                
                if output_audio is not None:
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        torchaudio.save(tmp_file.name, 
                                      torch.from_numpy(output_audio), 
                                      model.sample_rate)
                        return tmp_file.name, "✅ 음성이 생성되었습니다 (Zero-shot 모드)"
            
            return None, "❌ 음성 생성에 실패했습니다."
            
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return None, f"❌ 음성 생성 오류: {str(e)}"
    
    def prepare_reference_audio(self, audio_input):
        """참조 음성을 CosyVoice 형식으로 변환"""
        if isinstance(audio_input, str):
            # 파일 경로인 경우
            return load_wav(audio_input, 16000)
        else:
            # 튜플인 경우 (sample_rate, audio_data)
            sample_rate, audio_data = audio_input
            
            # 정규화 및 변환
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # 16kHz로 리샘플링
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, 
                                            orig_sr=sample_rate, 
                                            target_sr=16000)
            
            return torch.from_numpy(audio_data)
    
    def auto_generate(self, reference_audio, target_text, model_path, 
                     whisper_size="base", language="auto"):
        """자동 STT + TTS 파이프라인"""
        try:
            if reference_audio is None:
                return None, "❌ 참조 음성을 먼저 녹음해주세요."
            
            if not target_text.strip():
                return None, "❌ 생성할 텍스트를 입력해주세요."
            
            # 1. STT로 참조 텍스트 추출
            model = self.load_whisper(whisper_size)
            
            if isinstance(reference_audio, str):
                audio_path = reference_audio
            else:
                sample_rate, audio_data = reference_audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    
                    if sample_rate != 16000:
                        import librosa
                        audio_data = librosa.resample(audio_data, 
                                                    orig_sr=sample_rate, 
                                                    target_sr=16000)
                    
                    torchaudio.save(tmp_file.name, 
                                  torch.from_numpy(audio_data).unsqueeze(0), 
                                  16000)
                    audio_path = tmp_file.name
            
            # STT 실행
            if language == "auto":
                result = model.transcribe(audio_path)
            else:
                result = model.transcribe(audio_path, language=language)
            
            prompt_text = result["text"].strip()
            
            # 2. TTS로 음성 생성
            output_audio, message = self.generate_speech(
                reference_audio, prompt_text, target_text, model_path, "zero_shot"
            )
            
            # 임시 파일 정리
            if 'tmp_file' in locals():
                os.unlink(audio_path)
            
            if output_audio:
                return output_audio, f"✅ 자동 생성 완료!\n인식된 텍스트: '{prompt_text}'\n생성된 음성을 확인하세요."
            else:
                return None, f"❌ 자동 생성 실패: {message}"
            
        except Exception as e:
            logger.error(f"Auto generation error: {e}")
            return None, f"❌ 자동 생성 오류: {str(e)}"

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    app = CosyVoiceGradioApp()
    
    # 모델 경로 옵션
    model_options = [
        "pretrained_models/CosyVoice2-0.5B",
        "pretrained_models/CosyVoice-300M",
        "pretrained_models/CosyVoice-300M-SFT",
        "pretrained_models/CosyVoice-300M-Instruct"
    ]
    
    with gr.Blocks(title="🎙️ CosyVoice 음성 복제", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ CosyVoice 음성 복제 시스템")
        gr.Markdown("음성을 녹음하고 텍스트를 입력하면, 당신의 목소리로 음성을 생성합니다!")
        
        with gr.Tab("🚀 자동 생성 (추천)"):
            gr.Markdown("### 한 번에 음성 인식 → 음성 생성")
            
            with gr.Row():
                with gr.Column():
                    ref_audio_auto = gr.Audio(
                        label="📢 참조 음성 녹음", 
                        sources=["microphone"],
                        type="numpy"
                    )
                    target_text_auto = gr.Textbox(
                        label="📝 생성할 텍스트",
                        placeholder="여기에 생성하고 싶은 텍스트를 입력하세요...",
                        lines=3
                    )
                    
                    with gr.Row():
                        model_path_auto = gr.Dropdown(
                            choices=model_options,
                            value=model_options[0],
                            label="🤖 모델 선택"
                        )
                        whisper_size_auto = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium"],
                            value="base",
                            label="🎯 Whisper 모델"
                        )
                    
                    generate_btn = gr.Button("🎵 음성 생성하기", variant="primary", size="lg")
                
                with gr.Column():
                    output_audio_auto = gr.Audio(label="🔊 생성된 음성")
                    status_auto = gr.Textbox(label="📋 상태", interactive=False)
            
            generate_btn.click(
                fn=app.auto_generate,
                inputs=[ref_audio_auto, target_text_auto, model_path_auto, whisper_size_auto],
                outputs=[output_audio_auto, status_auto]
            )
        
        with gr.Tab("🔧 수동 설정"):
            gr.Markdown("### 단계별 음성 생성 (고급 사용자용)")
            
            with gr.Row():
                with gr.Column():
                    # STT 섹션
                    gr.Markdown("#### 1️⃣ 음성 인식 (STT)")
                    ref_audio_manual = gr.Audio(
                        label="📢 참조 음성 녹음", 
                        sources=["microphone"],
                        type="numpy"
                    )
                    
                    with gr.Row():
                        whisper_size_manual = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium"],
                            value="base",
                            label="Whisper 크기"
                        )
                        language = gr.Dropdown(
                            choices=["auto", "ko", "en", "zh", "ja"],
                            value="auto",
                            label="언어"
                        )
                    
                    transcribe_btn = gr.Button("🎯 음성 인식", variant="secondary")
                    transcribe_result = gr.Textbox(label="인식 결과", interactive=False)
                    
                    # TTS 섹션
                    gr.Markdown("#### 2️⃣ 음성 생성 (TTS)")
                    prompt_text_manual = gr.Textbox(
                        label="📝 참조 음성의 텍스트",
                        placeholder="위의 인식 결과를 복사하거나 직접 입력...",
                        lines=2
                    )
                    target_text_manual = gr.Textbox(
                        label="📝 생성할 텍스트",
                        placeholder="생성하고 싶은 텍스트를 입력하세요...",
                        lines=3
                    )
                    
                    with gr.Row():
                        model_path_manual = gr.Dropdown(
                            choices=model_options,
                            value=model_options[0],
                            label="🤖 모델"
                        )
                        mode = gr.Dropdown(
                            choices=["zero_shot", "sft"],
                            value="zero_shot",
                            label="모드"
                        )
                    
                    speaker_id = gr.Textbox(
                        label="화자 ID (SFT 모드용)",
                        value="중문여",
                        visible=False
                    )
                    
                    generate_manual_btn = gr.Button("🎵 음성 생성", variant="primary")
                
                with gr.Column():
                    output_audio_manual = gr.Audio(label="🔊 생성된 음성")
                    status_manual = gr.Textbox(label="📋 상태", interactive=False)
            
            # 모드 변경 시 화자 ID 입력 표시/숨김
            def update_speaker_visibility(mode_value):
                return gr.update(visible=(mode_value == "sft"))
            
            mode.change(
                fn=update_speaker_visibility,
                inputs=[mode],
                outputs=[speaker_id]
            )
            
            # 이벤트 바인딩
            transcribe_btn.click(
                fn=app.transcribe_audio,
                inputs=[ref_audio_manual, whisper_size_manual, language],
                outputs=[transcribe_result]
            )
            
            generate_manual_btn.click(
                fn=app.generate_speech,
                inputs=[ref_audio_manual, prompt_text_manual, target_text_manual, 
                       model_path_manual, mode, speaker_id],
                outputs=[output_audio_manual, status_manual]
            )
        
        with gr.Tab("ℹ️ 사용법"):
            gr.Markdown("""
            ## 📖 사용법
            
            ### 🚀 자동 생성 모드 (추천)
            1. **참조 음성 녹음**: 마이크 버튼을 클릭하여 5-10초간 음성을 녹음하세요
            2. **텍스트 입력**: 생성하고 싶은 텍스트를 입력하세요
            3. **모델 선택**: CosyVoice2-0.5B를 추천합니다 (성능 최고)
            4. **생성 클릭**: '음성 생성하기' 버튼을 클릭하세요
            
            ### 🔧 수동 설정 모드
            1. **STT**: 참조 음성 녹음 → 음성 인식 클릭
            2. **수정**: 인식 결과가 틀렸다면 수동으로 수정
            3. **TTS**: 생성할 텍스트 입력 → 음성 생성 클릭
            
            ## ⚠️ 주의사항
            - **참조 음성**: 5-10초, 명확한 발음, 노이즈 최소화
            - **지원 언어**: 한국어, 영어, 중국어, 일본어
            - **모델 크기**: 처음 실행 시 모델 다운로드로 시간 소요
            
            ## 🔧 모델 설명
            - **CosyVoice2-0.5B**: 최신 모델, 성능 최고 (추천)
            - **CosyVoice-300M-SFT**: 미리 학습된 화자들 사용
            - **CosyVoice-300M-Instruct**: 감정/스타일 제어 가능
            """)
    
    return demo

if __name__ == "__main__":
    # 모델 경로 확인
    model_dir = Path("pretrained_models")
    if not model_dir.exists():
        print("⚠️ pretrained_models 폴더가 없습니다.")
        print("다음 명령어로 모델을 다운로드하세요:")
        print("python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')\"")
    
    # Gradio 앱 실행
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )