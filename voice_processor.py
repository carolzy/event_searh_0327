import os
import logging
import io
import tempfile
import wave
from typing import Dict, Any, Optional
from datetime import datetime
from base64 import b64encode
import speech_recognition as sr
from pydub import AudioSegment
from elevenlabs import generate, set_api_key, Voice, VoiceSettings, Model
import asyncio
import aiohttp
import base64
import json
import uuid
from pathlib import Path
import shutil
import time
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnboardingManager:
    """Manages the onboarding process"""
    
    def __init__(self, flow_controller):
        self.flow = flow_controller

class VoiceProcessor:
    """Handles voice processing, including transcription and text-to-speech"""
    
    def __init__(self, flow_controller=None):
        """Initialize the voice processor"""
        self.flow = flow_controller
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory at {self.temp_dir}")
        
        # Get ElevenLabs API key from environment variable
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.tts_enabled = self.elevenlabs_api_key is not None
        
        if not self.elevenlabs_api_key:
            logger.warning("ElevenLabs API key not found. Voice features will be disabled.")
        else:
            # Set the API key for ElevenLabs
            set_api_key(self.elevenlabs_api_key)
            logger.info("ElevenLabs API key set successfully.")
        
        # Initialize speech recognizer as fallback
        self.recognizer = sr.Recognizer()
        
        # Initialize onboarding manager only if flow_controller is provided
        if flow_controller:
            self.onboarding_manager = OnboardingManager(flow_controller)
        
        # B2B target companies by sector
        self.target_companies = {
            'Technology': ['Microsoft', 'Databricks', 'Salesforce', 'Oracle', 'SAP'],
            'Finance': ['Stripe', 'Square', 'Plaid', 'Adyen', 'Brex'],
            'Healthcare': ['Epic Systems', 'Cerner', 'Veeva', 'Athenahealth'],
            'Manufacturing': ['Siemens', 'Rockwell Automation', 'PTC', 'Autodesk']
        }
        
        # Default voice settings
        self.voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL')  # Default to "Bella" voice

    async def save_temp_audio(self, audio_file):
        """Save uploaded audio file to temporary directory"""
        filename = f"{uuid.uuid4()}.webm"
        filepath = os.path.join(self.temp_dir, filename)
        await audio_file.save(filepath)
        return filepath
    
    async def transcribe_audio(self, audio_path):
        """Transcribe audio file using ElevenLabs Scribe API and fall back to Google STT if necessary"""
        logger.info(f"Transcribing audio from {audio_path}")
        
        try:
            # Check if ElevenLabs API key is available
            if not self.elevenlabs_api_key:
                logger.error("ElevenLabs API key not found. Cannot transcribe audio.")
                return None
                
            # Determine the file extension and convert if necessary
            file_ext = os.path.splitext(audio_path)[1].lower()
            mp3_path = os.path.splitext(audio_path)[0] + '.mp3'
            
            logger.info(f"Audio file format: {file_ext}")
            logger.info(f"Original audio path: {audio_path}")
            logger.info(f"Target MP3 path: {mp3_path}")
            
            # Convert to mp3 if not already in that format
            if file_ext != '.mp3':
                try:
                    logger.info(f"Converting {file_ext} to mp3 format")
                    audio_segment = AudioSegment.from_file(audio_path)
                    audio_segment.export(mp3_path, format="mp3")
                    logger.info(f"Converted {file_ext} to mp3: {mp3_path}")
                except Exception as e:
                    logger.error(f"Error converting audio to mp3: {str(e)}")
                    return await self._fallback_transcription(audio_path)
            else:
                mp3_path = audio_path
                logger.info("Audio already in mp3 format, no conversion needed")
            
            # Read the audio file
            try:
                with open(mp3_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                logger.info(f"Successfully read audio file, size: {len(audio_data)} bytes")
            except Exception as e:
                logger.error(f"Error reading audio file: {str(e)}")
                return await self._fallback_transcription(audio_path)
            
            # Prepare the request to ElevenLabs Scribe API
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            # Create form data with the audio file
            form_data = aiohttp.FormData()
            form_data.add_field('file', 
                               audio_data, 
                               filename='audio.mp3', 
                               content_type='audio/mpeg')
            form_data.add_field('model_id', 'scribe_v1')
            
            logger.info("Prepared request to ElevenLabs Scribe API")
            
            # Make the API request
            try:
                logger.info("Sending request to ElevenLabs Scribe API")
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=form_data, headers=headers) as response:
                        logger.info(f"Received response from ElevenLabs, status: {response.status}")
                        if response.status == 200:
                            result = await response.json()
                            transcription = result.get('text', '')
                            logger.info(f"ElevenLabs transcription: {transcription}")
                            return transcription
                        elif response.status == 401 and "quota_exceeded" in await response.text():
                            logger.warning("ElevenLabs quota exceeded. Using fallback text-to-speech.")
                            return None
                        else:
                            error_text = await response.text()
                            logger.error(f"Error from ElevenLabs Scribe API: {error_text}")
                            # Fall back to Google STT
                            return await self._fallback_transcription(audio_path)
            except Exception as e:
                logger.error(f"Error making request to ElevenLabs: {str(e)}")
                # Fall back to Google STT
                return await self._fallback_transcription(audio_path)
            
        except Exception as e:
            logger.error(f"Error transcribing audio with ElevenLabs: {str(e)}")
            # Fall back to Google STT
            return await self._fallback_transcription(audio_path)
        finally:
            # Clean up the temporary files
            try:
                if mp3_path != audio_path and os.path.exists(mp3_path):
                    os.remove(mp3_path)
                    logger.info(f"Removed temporary mp3 file: {mp3_path}")
                
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Removed original audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}")
                
    async def _fallback_transcription(self, audio_path):
        """Fall back to Google STT if ElevenLabs transcription fails"""
        try:
            logger.info("Attempting fallback to Google STT...")
            
            # Convert to wav for Google STT
            wav_path = os.path.splitext(audio_path)[0] + '.wav'
            
            try:
                # Convert to WAV format for Google STT
                audio_segment = AudioSegment.from_file(audio_path)
                audio_segment.export(wav_path, format="wav")
                logger.info(f"Converted to wav for Google STT: {wav_path}")
            except Exception as e:
                logger.error(f"Error converting to wav for Google STT: {str(e)}")
                return None
                
            # Use Google's Speech Recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                
            transcription = recognizer.recognize_google(audio_data)
            logger.info(f"Google STT transcription: {transcription}")
            
            # Clean up temporary wav file
            if os.path.exists(wav_path):
                os.remove(wav_path)
                logger.info(f"Removed temporary wav file: {wav_path}")
                
            return transcription
        except Exception as e:
            logger.error(f"Fallback transcription also failed: {str(e)}")
            return None
    
    async def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs API"""
        if not self.elevenlabs_api_key or not text:
            logger.warning(f"TTS skipped - API key missing: {not self.elevenlabs_api_key}, Text empty: {not text}")
            return None
        
        try:
            # Log the full text being sent to ElevenLabs
            logger.info(f"Generating TTS for full text: {text}")
            
            # Use the API directly with httpx instead of the elevenlabs library
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    audio_data = response.content
                    # Convert to base64 for sending over JSON
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    logger.info(f"TTS generated successfully, audio size: {len(audio_data)} bytes")
                    return audio_b64
                elif response.status_code == 401 and "quota_exceeded" in response.text:
                    logger.warning("ElevenLabs quota exceeded. Using fallback text-to-speech.")
                    return "QUOTA_EXCEEDED"
                else:
                    logger.error(f"Error from ElevenLabs API: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None
    
    async def process_onboarding_step(self, step, text, context):
        """Process an onboarding step"""
        from question_engine import QuestionEngine
        
        # Initialize the question engine
        question_engine = QuestionEngine()
        
        # Check if this is the last step
        is_last_step = step == 'additional'
        next_step = None if is_last_step else self._get_next_step(step)
        
        # Generate the next question
        next_question = question_engine.generate(next_step, context)
        
        # Convert to speech if enabled
        audio_data = await self.text_to_speech(next_question) if next_question else None
        
        return {
            "text": next_question,
            "audio": audio_data,
            "completed": is_last_step
        }
    
    def _get_next_step(self, current_step):
        """Get the next step in the onboarding flow"""
        steps = ['product', 'market', 'company_size', 'additional']
        
        try:
            current_index = steps.index(current_step)
            if current_index < len(steps) - 1:
                return steps[current_index + 1]
        except ValueError:
            pass
        
        return None
        
    async def speech_to_text(self, audio_data: bytes) -> str:
        """Transcribe audio to text using ElevenLabs Scribe API"""
        try:
            # Save audio data to a temporary file
            temp_file = os.path.join(self.temp_dir, f"{uuid.uuid4()}.webm")
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
                
            # Transcribe the audio
            transcription = await self.transcribe_audio(temp_file)
            if transcription:
                return transcription
                
            return ""
                     
        except Exception as e:
            logger.error(f"Error in speech to text: {str(e)}")
            return ""
            
    async def process_voice_command(self, audio_data: bytes) -> Dict[str, Any]:
        """Process voice command and manage conversation flow"""
        try:
            # Transcribe audio to text
            command = await self.speech_to_text(audio_data)
            
            if not command:
                return {
                    "error": "Could not understand audio. Please try again."
                }
                
            # Process the command based on the flow state
            response_text = f"You said: {command}"
            
            # Generate audio response if TTS is enabled
            audio_data = await self.text_to_speech(response_text) if self.tts_enabled else None
            
            return {
                "text": response_text,
                "audio": audio_data
            }
            
        except Exception as e:
            logger.error(f"Error processing voice command: {str(e)}")
            return {"error": str(e)}
            
    def __del__(self):
        """Clean up temporary directory when the object is destroyed"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory {self.temp_dir}")
