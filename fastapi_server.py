# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import io
import base64
import tempfile
from typing import Optional
import uvicorn
import numpy as np
import torch
import torchaudio
import librosa
import whisper
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import soundfile as sf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI(
    title="CosyVoice Zero-Shot TTS API with Whisper",
    description="""
    FastAPI server for CosyVoice zero-shot voice cloning with automatic transcription using Whisper.
    
    ## Features
    - **Zero-shot TTS**: Generate speech in any voice using just a short audio sample
    - **Automatic Transcription**: Uses Whisper to automatically transcribe prompt audio
    - **Multiple Input Methods**: Support for base64 encoded audio and file uploads
    - **Streaming Support**: Real-time audio generation
    - **High Quality**: Advanced audio processing and validation
    
    ## Usage
    1. Provide text to synthesize
    2. Upload or send prompt audio (the voice you want to clone)
    3. The system automatically transcribes the prompt audio using Whisper
    4. Generates speech in the target voice
    
    ## Audio Requirements
    - Minimum sample rate: 16kHz
    - Supported formats: WAV, MP3, etc.
    - Recommended length: 3-30 seconds for best results
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Global variables
cosyvoice = None
whisper_model = None
prompt_sr = 16000
max_val = 0.8

class TTSRequest(BaseModel):
    text: str = Field(
        ..., 
        description="Text to synthesize into speech",
        example="안녕하세요, 저는 인공지능 음성 합성 시스템입니다."
    )
    prompt_audio_base64: str = Field(
        ..., 
        description="Base64 encoded prompt audio file (WAV format recommended)",
        example="UklGRiQAAABXQVZFZm10IBAAAAABAAEATECAA..."
    )
    seed: int = Field(
        42, 
        description="Random seed for reproducible generation", 
        ge=1, 
        le=100000000,
        example=12345
    )
    stream: bool = Field(
        False, 
        description="Enable streaming response (collects all chunks before returning)"
    )
    speed: float = Field(
        1.0, 
        description="Speed adjustment factor for speech generation", 
        ge=0.5, 
        le=2.0,
        example=1.0
    )

class TTSResponse(BaseModel):
    success: bool = Field(description="Whether the request was successful")
    message: str = Field(description="Status message")
    transcribed_text: str = Field(description="Text transcribed from prompt audio using Whisper")
    audio_base64: Optional[str] = Field(None, description="Generated audio as base64 encoded WAV")
    sample_rate: int = Field(description="Sample rate of generated audio")

class TranscriptionResponse(BaseModel):
    success: bool = Field(description="Whether transcription was successful")
    transcribed_text: str = Field(description="Text transcribed from audio")
    message: str = Field(description="Status message")

class HealthResponse(BaseModel):
    status: str = Field(description="Service health status")
    cosyvoice_loaded: bool = Field(description="Whether CosyVoice model is loaded")
    whisper_loaded: bool = Field(description="Whether Whisper model is loaded")
    sample_rate: Optional[int] = Field(description="Audio sample rate used by CosyVoice")

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """Post-process the generated speech"""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def audio_to_base64(audio_data, sample_rate):
    """Convert audio numpy array to base64 encoded WAV"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
    return audio_base64

def base64_to_audio(audio_base64):
    """Convert base64 encoded audio to file path"""
    try:
        audio_data = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")
        
        result = whisper_model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with basic service information"""
    return {"message": "CosyVoice Zero-Shot TTS API with Whisper", "status": "running"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint to verify service and model status"""
    return HealthResponse(
        status="healthy",
        cosyvoice_loaded=cosyvoice is not None,
        whisper_loaded=whisper_model is not None,
        sample_rate=cosyvoice.sample_rate if cosyvoice else None
    )

@app.post("/tts/zero_shot", response_model=TTSResponse, tags=["Text-to-Speech"])
async def zero_shot_tts(request: TTSRequest):
    """
    Generate speech using zero-shot voice cloning with automatic transcription.
    
    This endpoint takes text to synthesize and a base64-encoded audio file as a voice prompt.
    The system automatically transcribes the prompt audio using Whisper, then generates 
    speech in the target voice.
    
    **Process:**
    1. Decode the base64 prompt audio
    2. Transcribe the audio using Whisper to get prompt text
    3. Generate speech in the prompt voice using CosyVoice
    4. Return the generated audio as base64
    
    **Audio Requirements:**
    - Format: WAV recommended (supports most audio formats)
    - Sample rate: Minimum 16kHz
    - Duration: 3-30 seconds for best results
    - Quality: Clear speech with minimal background noise
    """
    try:
        if not cosyvoice:
            raise HTTPException(status_code=500, detail="CosyVoice model not loaded")
        
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")

        if not request.prompt_audio_base64:
            raise HTTPException(status_code=400, detail="Prompt audio is required for zero-shot TTS")

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="TTS text cannot be empty")

        # Convert base64 audio to temporary file
        prompt_wav_path = base64_to_audio(request.prompt_audio_base64)
        
        try:
            # Check audio sample rate
            audio_info = torchaudio.info(prompt_wav_path)
            if audio_info.sample_rate < prompt_sr:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Prompt audio sample rate {audio_info.sample_rate} is below required {prompt_sr}Hz"
                )

            # Transcribe prompt audio using Whisper
            prompt_text = transcribe_audio(prompt_wav_path)
            
            if not prompt_text.strip():
                raise HTTPException(status_code=400, detail="Failed to transcribe prompt audio")

            # Process prompt audio
            prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
            
            # Set random seed
            set_all_random_seed(request.seed)
            
            logging.info(f'Processing zero_shot inference - TTS text: "{request.text}", Transcribed prompt: "{prompt_text}"')
            
            if request.stream:
                # For streaming, collect all chunks
                audio_chunks = []
                for chunk in cosyvoice.inference_zero_shot(
                    request.text, 
                    prompt_text, 
                    prompt_speech_16k, 
                    stream=True, 
                    speed=request.speed
                ):
                    audio_chunks.append(chunk['tts_speech'].numpy().flatten())
                
                final_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
            else:
                # Non-streaming inference
                result = next(cosyvoice.inference_zero_shot(
                    request.text,
                    prompt_text,
                    prompt_speech_16k,
                    stream=False,
                    speed=request.speed
                ))
                final_audio = result['tts_speech'].numpy().flatten()
            
            # Convert to base64
            audio_base64 = audio_to_base64(final_audio, cosyvoice.sample_rate)
            
            return TTSResponse(
                success=True,
                message="TTS generation completed successfully",
                transcribed_text=prompt_text,
                audio_base64=audio_base64,
                sample_rate=cosyvoice.sample_rate
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(prompt_wav_path):
                os.unlink(prompt_wav_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts/zero_shot_file", tags=["Text-to-Speech"])
async def zero_shot_tts_file(
    text: str = Form(..., description="Text to synthesize into speech"),
    prompt_audio: UploadFile = File(..., description="Audio file containing the voice to clone"),
    seed: int = Form(42, description="Random seed for reproducible generation"),
    stream: bool = Form(False, description="Enable streaming response"),
    speed: float = Form(1.0, description="Speed adjustment factor (0.5-2.0)")
):
    """
    Generate speech using zero-shot voice cloning with file upload.
    
    This endpoint accepts an audio file upload as the voice prompt and automatically
    transcribes it using Whisper before generating speech in the target voice.
    
    **Returns:** Audio file as streaming response with transcribed text in headers.
    
    **Process:**
    1. Upload audio file containing the target voice
    2. Transcribe the audio using Whisper
    3. Generate speech in the prompt voice
    4. Return audio file directly
    
    **Audio Requirements:**
    - Supported formats: WAV, MP3, FLAC, etc.
    - Sample rate: Minimum 16kHz  
    - Duration: 3-30 seconds recommended
    - Quality: Clear speech, minimal noise
    """
    try:
        if not cosyvoice:
            raise HTTPException(status_code=500, detail="CosyVoice model not loaded")
        
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")

        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="TTS text cannot be empty")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await prompt_audio.read()
            temp_file.write(content)
            prompt_wav_path = temp_file.name

        try:
            # Check audio sample rate
            audio_info = torchaudio.info(prompt_wav_path)
            if audio_info.sample_rate < prompt_sr:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt audio sample rate {audio_info.sample_rate} is below required {prompt_sr}Hz"
                )

            # Transcribe prompt audio using Whisper
            prompt_text = transcribe_audio(prompt_wav_path)
            
            if not prompt_text.strip():
                raise HTTPException(status_code=400, detail="Failed to transcribe prompt audio")

            # Process prompt audio
            prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
            
            # Set random seed
            set_all_random_seed(seed)
            
            logging.info(f'Processing zero_shot file inference - TTS text: "{text}", Transcribed prompt: "{prompt_text}"')
            
            if stream:
                # For streaming, collect all chunks
                audio_chunks = []
                for chunk in cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=True, speed=speed):
                    audio_chunks.append(chunk['tts_speech'].numpy().flatten())
                final_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
            else:
                # Non-streaming inference
                result = next(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False, speed=speed))
                final_audio = result['tts_speech'].numpy().flatten()
            
            # Return audio as streaming response
            buffer = io.BytesIO()
            sf.write(buffer, final_audio, cosyvoice.sample_rate, format='WAV')
            buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(buffer.getvalue()),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=generated_audio.wav",
                    "X-Transcribed-Text": prompt_text
                }
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(prompt_wav_path):
                os.unlink(prompt_wav_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Speech Recognition"])
async def transcribe_only(prompt_audio: UploadFile = File(..., description="Audio file to transcribe")):
    """
    Transcribe audio file using Whisper speech recognition.
    
    This endpoint only performs transcription without TTS generation. Useful for:
    - Testing audio quality before TTS
    - Getting transcription for verification
    - Speech-to-text applications
    
    **Supported formats:** WAV, MP3, FLAC, M4A, etc.
    **Languages:** Automatic detection (supports 99+ languages)
    """
    try:
        if not whisper_model:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await prompt_audio.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Transcribe audio
            transcribed_text = transcribe_audio(temp_path)
            
            return TranscriptionResponse(
                success=True,
                transcribed_text=transcribed_text,
                message="Transcription completed successfully"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def init_models(model_dir: str, whisper_model_name: str = "base"):
    """Initialize CosyVoice and Whisper models"""
    global cosyvoice, whisper_model
    
    # Initialize CosyVoice
    try:
        cosyvoice = CosyVoice(model_dir)
        logging.info(f"Successfully loaded CosyVoice model from {model_dir}")
    except Exception:
        try:
            cosyvoice = CosyVoice2(model_dir)
            logging.info(f"Successfully loaded CosyVoice2 model from {model_dir}")
        except Exception as e:
            logging.error(f"Failed to load CosyVoice model from {model_dir}: {str(e)}")
            raise TypeError('No valid CosyVoice model found!')
    
    # Initialize Whisper
    try:
        whisper_model = whisper.load_model(whisper_model_name)
        logging.info(f"Successfully loaded Whisper model: {whisper_model_name}")
    except Exception as e:
        logging.error(f"Failed to load Whisper model {whisper_model_name}: {str(e)}")
        raise TypeError(f'Failed to load Whisper model: {str(e)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice Zero-Shot TTS FastAPI Server with Whisper")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', 
                       help='Local path or modelscope repo id for CosyVoice')
    parser.add_argument('--whisper_model', type=str, default='base',
                       help='Whisper model name (tiny, base, small, medium, large)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Initialize models
    init_models(args.model_dir, args.whisper_model)
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="CosyVoice Zero-Shot TTS API with Whisper",
            version="1.0.0",
            description=app.description,
            routes=app.routes,
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://img.shields.io/badge/CosyVoice-TTS-blue"
        }
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
    
    # Run server
    print(f"🚀 Starting CosyVoice TTS API Server")
    print(f"📝 Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"📋 ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"🔧 OpenAPI JSON: http://{args.host}:{args.port}/openapi.json")
    print(f"🎤 Whisper Model: {args.whisper_model}")
    print(f"🎯 CosyVoice Model: {args.model_dir}")
    
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )