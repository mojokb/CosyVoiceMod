#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import requests
import base64
import io
import tempfile
import os
from pathlib import Path
import soundfile as sf

FASTAPI_BASE_URL = "http://localhost:8000"

def check_server_health():
    """Check if FastAPI server is running"""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("status") == "healthy"
    except:
        pass
    return False

def audio_file_to_base64(audio_file_path):
    """Convert audio file to base64 string"""
    if audio_file_path is None:
        return None
    
    with open(audio_file_path, "rb") as f:
        audio_data = f.read()
    
    return base64.b64encode(audio_data).decode()

def base64_to_audio_file(audio_base64, sample_rate=22050):
    """Convert base64 audio to temporary file and return path"""
    if not audio_base64:
        return None
    
    try:
        audio_data = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    except Exception as e:
        print(f"Error converting base64 to audio: {e}")
        return None

def generate_speech(text, prompt_audio, seed=42, speed=1.0, stream=False):
    """Generate speech using FastAPI backend"""
    
    if not check_server_health():
        return None, "❌ FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
    
    if not text or not text.strip():
        return None, "❌ 텍스트를 입력해주세요."
    
    if prompt_audio is None:
        return None, "❌ 음성 프롬프트 파일을 업로드해주세요."
    
    try:
        # Convert audio file to base64
        audio_base64 = audio_file_to_base64(prompt_audio)
        if not audio_base64:
            return None, "❌ 음성 파일을 읽을 수 없습니다."
        
        # Prepare request data
        request_data = {
            "text": text,
            "prompt_audio_base64": audio_base64,
            "seed": int(seed),
            "stream": stream,
            "speed": float(speed)
        }
        
        # Make API request
        response = requests.post(
            f"{FASTAPI_BASE_URL}/tts/zero_shot",
            json=request_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                # Convert base64 audio back to file
                generated_audio_path = base64_to_audio_file(
                    result.get("audio_base64"),
                    result.get("sample_rate", 22050)
                )
                
                transcribed_text = result.get("transcribed_text", "")
                message = f"✅ 음성 생성 완료!\n📝 인식된 프롬프트 텍스트: \"{transcribed_text}\""
                
                return generated_audio_path, message
            else:
                return None, f"❌ 서버 오류: {result.get('message', '알 수 없는 오류')}"
        
        else:
            error_detail = "알 수 없는 오류"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", str(error_data))
            except:
                error_detail = response.text
            
            return None, f"❌ API 오류 ({response.status_code}): {error_detail}"
    
    except requests.exceptions.Timeout:
        return None, "❌ 요청 시간 초과. 서버 응답이 너무 늦습니다."
    except requests.exceptions.ConnectionError:
        return None, "❌ 서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요."
    except Exception as e:
        return None, f"❌ 예상치 못한 오류: {str(e)}"

def transcribe_audio(audio_file):
    """Transcribe audio using FastAPI backend"""
    
    if not check_server_health():
        return "❌ FastAPI 서버에 연결할 수 없습니다."
    
    if audio_file is None:
        return "❌ 음성 파일을 업로드해주세요."
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'prompt_audio': ('audio.wav', f, 'audio/wav')}
            response = requests.post(
                f"{FASTAPI_BASE_URL}/transcribe",
                files=files,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return f"📝 인식된 텍스트: \"{result.get('transcribed_text', '')}\""
            else:
                return f"❌ 전사 실패: {result.get('message', '알 수 없는 오류')}"
        else:
            return f"❌ API 오류 ({response.status_code})"
    
    except Exception as e:
        return f"❌ 오류: {str(e)}"

# Gradio Interface
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="CosyVoice TTS with Whisper",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 0.5rem;
            margin: 1.5rem 0 1rem 0;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            <div class="main-header">
            
            # 🎤 CosyVoice Zero-Shot TTS
            
            **Whisper 자동 전사 기능이 포함된 음성 복제 시스템**
            
            음성 프롬프트를 업로드하면 Whisper가 자동으로 텍스트를 인식하여 해당 목소리로 새로운 텍스트를 음성 합성합니다.
            
            </div>
            """, 
            elem_classes=["main-header"]
        )
        
        # Server status
        with gr.Row():
            server_status = gr.HTML()
            
            def update_server_status():
                if check_server_health():
                    return '<div style="color: green; font-weight: bold;">🟢 서버 연결 상태: 정상</div>'
                else:
                    return '<div style="color: red; font-weight: bold;">🔴 서버 연결 상태: 연결 실패</div>'
            
            server_status.value = update_server_status()
        
        with gr.Tab("🎯 음성 생성 (Zero-Shot TTS)"):
            gr.Markdown("### 📝 합성할 텍스트", elem_classes=["section-header"])
            
            text_input = gr.Textbox(
                label="텍스트 입력",
                placeholder="합성하고 싶은 텍스트를 입력하세요... (예: 안녕하세요, 저는 인공지능 음성 합성 시스템입니다.)",
                lines=3,
                max_lines=5
            )
            
            gr.Markdown("### 🎤 음성 프롬프트", elem_classes=["section-header"])
            
            prompt_audio = gr.Audio(
                label="목소리 샘플 업로드",
                type="filepath",
                format="wav"
            )
            
            gr.Markdown(
                """
                **📋 음성 프롬프트 요구사항:**
                - 형식: WAV, MP3 등 지원
                - 샘플링 레이트: 최소 16kHz
                - 길이: 3-30초 권장
                - 품질: 명확한 발음, 배경소음 최소화
                """
            )
            
            gr.Markdown("### ⚙️ 생성 옵션", elem_classes=["section-header"])
            
            with gr.Row():
                seed_input = gr.Number(
                    label="시드값",
                    value=42,
                    minimum=1,
                    maximum=100000000,
                    step=1,
                    info="동일한 시드값으로 재현 가능한 결과"
                )
                
                speed_input = gr.Slider(
                    label="속도 조절",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    info="말하기 속도 (0.5=느리게, 2.0=빠르게)"
                )
                
                stream_input = gr.Checkbox(
                    label="스트리밍 모드",
                    value=False,
                    info="실시간 생성 (실험적 기능)"
                )
            
            generate_btn = gr.Button(
                "🎵 음성 생성",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("### 🔊 결과", elem_classes=["section-header"])
            
            with gr.Row():
                output_audio = gr.Audio(
                    label="생성된 음성",
                    type="filepath"
                )
            
            result_message = gr.Textbox(
                label="상태 메시지",
                interactive=False,
                lines=3
            )
            
            # Generate button event
            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, prompt_audio, seed_input, speed_input, stream_input],
                outputs=[output_audio, result_message],
                show_progress=True
            )
        
        with gr.Tab("📝 음성 전사 (Whisper)"):
            gr.Markdown(
                """
                ### 🎤 음성 파일 전사
                
                업로드한 음성 파일을 Whisper를 사용하여 텍스트로 변환합니다.
                TTS 생성 전에 음성 품질을 확인하거나 단순히 음성을 텍스트로 변환할 때 사용하세요.
                """
            )
            
            transcribe_audio_input = gr.Audio(
                label="전사할 음성 파일",
                type="filepath"
            )
            
            transcribe_btn = gr.Button("📝 음성 전사", variant="secondary")
            
            transcribe_result = gr.Textbox(
                label="전사 결과",
                interactive=False,
                lines=4
            )
            
            # Transcribe button event
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[transcribe_audio_input],
                outputs=[transcribe_result],
                show_progress=True
            )
        
        with gr.Tab("ℹ️ 정보"):
            gr.Markdown(
                """
                ## 📖 사용 방법
                
                ### 1️⃣ 음성 생성
                1. **텍스트 입력**: 합성하고 싶은 텍스트를 입력
                2. **음성 프롬프트 업로드**: 복제하고 싶은 목소리의 음성 파일 업로드
                3. **옵션 설정**: 시드값, 속도 등 조정 (선택사항)
                4. **생성 버튼 클릭**: 음성 생성 시작
                
                ### 2️⃣ 음성 전사
                - 음성 파일을 업로드하여 텍스트로 변환
                - TTS 생성 전 음성 품질 확인 용도로 활용
                
                ## 🔧 기술 정보
                
                - **CosyVoice**: 알리바바에서 개발한 고품질 음성 합성 모델
                - **Whisper**: OpenAI의 자동 음성 인식 모델  
                - **Zero-Shot TTS**: 사전 학습 없이 새로운 목소리로 음성 합성
                - **FastAPI Backend**: 고성능 REST API 서버
                
                ## ⚠️ 주의사항
                
                - 음성 프롬프트는 명확하고 깨끗한 음성을 사용하세요
                - 저작권이 있는 음성이나 타인의 동의 없는 음성 사용을 피하세요
                - 생성된 음성을 악용하지 마세요
                
                ## 🛠️ 문제 해결
                
                - **서버 연결 실패**: FastAPI 서버가 실행 중인지 확인 (`python fastapi_server.py`)
                - **음성 품질 낮음**: 더 긴 시간(10-30초)의 깨끗한 프롬프트 사용
                - **생성 실패**: 음성 파일 형식과 샘플링 레이트 확인
                """
            )
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    
    print("🚀 CosyVoice Gradio 앱을 시작합니다...")
    print("📡 FastAPI 서버 상태 확인 중...")
    
    if check_server_health():
        print("✅ FastAPI 서버 연결 성공!")
    else:
        print("❌ FastAPI 서버에 연결할 수 없습니다.")
        print("   다음 명령으로 서버를 먼저 실행하세요:")
        print("   python fastapi_server.py")
    
    # Launch the app
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )