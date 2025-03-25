import os
import re
import csv
import json
import time
import uuid
import base64
import random
import logging
import asyncio
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
import google.generativeai as genai
from quart import Quart, render_template, request, jsonify, websocket, redirect, url_for, send_file, make_response, send_from_directory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Google Generative AI
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# Initialize the Flask app
app = Quart(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Force template reloading

# Generate a version timestamp to bust cache
VERSION = str(int(time.time()))

# Initialize global variables
flow_controller = None
voice_processor = None
question_engine = None
company_recommender = None

# Global conversation history
conversation_history = []

# Common English stopwords to filter out from keywords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 
    'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 
    'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from', 'in', 
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
    'same', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'also', 'been',
    'have', 'has', 'had', 'would', 'could', 'with', 'they', 'their', 'them', 'we',
    'our', 'us', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'i', 'me', 'my', 'mine', 'am', 'are', 'was', 'were', 'be', 'being',
    'been', 'do', 'does', 'did', 'doing', 'get', 'gets', 'got', 'getting'
}

# Application settings
settings = {
    'tts_enabled': True,
    'debug_mode': False,
    'max_recommendations': 5,
    'recommendation_threshold': 0.5
}

class VoiceProcessor:
    """Class for handling voice processing operations."""
    
    def __init__(self, flow_controller=None):
        """Initialize the voice processor."""
        # Use the new API key directly
        self.elevenlabs_api_key = "sk_7c40da2fa324cd316f1c891b942b076af56366f3b0367323"
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID
        self.flow_controller = flow_controller
    
    async def generate_speech(self, text):
        """Generate speech from text using ElevenLabs API."""
        if not text:
            logger.error("Empty text provided for speech generation")
            return None
        
        if not self.elevenlabs_api_key:
            logger.error("ElevenLabs API key not found")
            return None
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            async with httpx.AsyncClient() as session:
                response = await session.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    audio_data = response.content  # This is bytes, don't await it
                    # Convert to base64 for sending over JSON
                    base64_audio = base64.b64encode(audio_data).decode('utf-8')
                    logger.info(f"Generated speech for text: {text[:50]}...")
                    return base64_audio
                elif response.status_code == 429:
                    logger.error(f"ElevenLabs API quota exceeded")
                    return "QUOTA_EXCEEDED"
                else:
                    error_text = response.text  # This is a property, don't await it
                    logger.error(f"Error generating speech: {response.status_code} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Exception in generate_speech: {str(e)}")
            return None
    
    async def text_to_speech(self, text):
        """Alias for generate_speech for backward compatibility."""
        return await self.generate_speech(text)

class KeywordGenerator:
    async def _generate_keywords_with_llm(self, context):
        """Generate keywords based on the conversation context."""
        # Extract user input from context
        user_input = context.strip() if context else "cleanlab.ai"
        
        # Improved prompt that uses the user input and explicitly requests format
        prompt = f"""Based on the following user context, generate sales targeting related keywords:

USER CONTEXT:
{user_input}

Generate as many keywords as possible that would be highly relevant for B2B sales targeting based on:
1. The product/service mentioned
2. The industry/market segments  
3. Potential use cases
4. Common pain points this would solve
5. Related technologies or solutions

Return ONLY a comma-separated list of keywords without any explanations, categories, or formatting.
Include specific technical terms, industry jargon, and high-value targeting keywords that sales professionals would use.
Do not limit the number of keywords - generate as many relevant keywords as possible."""
        
        try:
            # Use the Gemini API directly
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("No Gemini API key found")
                return ["Sales", "Marketing", "Technology", "Innovation", "Business Development"]
                
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.3,  # Lower temperature for more focused keywords
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 4096  # Increased token limit for more keywords
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            logger.info(f"Generating keywords for context: {user_input[:50]}...")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=data,
                    timeout=15.0
                )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        keywords_text = content["parts"][0]["text"].strip()
                        # Split by commas and clean up
                        keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
                        
                        # Filter out stopwords but don't limit the number
                        filtered_keywords = [k for k in keywords if k.lower() not in STOPWORDS]
                        
                        # Log all generated keywords
                        logger.info(f"Generated {len(filtered_keywords)} keywords: {filtered_keywords}")
                        
                        return filtered_keywords
            logger.error(f"Error or unexpected response from Gemini API: {response.status_code} - {response.text}")
            return ["Sales", "Marketing", "Technology", "Innovation", "Business Development"]
        except Exception as e:
            logger.error(f"Error generating keywords with LLM: {str(e)}")
            # Return some default keywords if generation fails
            return ["Sales", "Marketing", "Technology", "Innovation", "Business Development"]

app.keyword_generator = KeywordGenerator()

@app.route('/')
async def index():
    """Serve the main page"""
    try:
        # Generate a natural greeting using Gemini
        greeting = await generate_natural_greeting()
        
        # Get the flow controller instance
        global flow_controller
        if flow_controller is None:
            flow_controller = FlowController()
        
        # Reset the flow controller
        flow_controller.current_step = 'product'
        flow_controller.current_product_line = ''
        flow_controller.current_sector = ''
        flow_controller.current_segment = ''
        flow_controller.keywords = []
        flow_controller.conversation_memory = []
        
        # Get the first question
        first_question = await flow_controller.generate_personalized_response('product')
        
        return await render_template('index.html', 
                                    greeting=greeting,
                                    first_question=first_question,
                                    version=VERSION)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return str(e), 500

async def generate_natural_greeting():
    """Generate a natural greeting using Gemini 2.0 Flash."""
    try:
        # Create prompt for greeting
        prompt = """Generate a friendly, conversational welcome message for a B2B sales assistant named Atom.ai.
        The message should:
        1. Introduce Atom.ai as a helpful assistant for B2B sales needs
        2. Mention that it can help identify potential companies to target
        3. Sound natural and conversational, not robotic
        4. Be brief (1-2 sentences)
        5. End with a question about what product the user sells
        
        The greeting MUST include the name "Atom.ai" with this exact capitalization.
        """
        
        logger.info("Generating natural greeting with Gemini 2.0 Flash")
        greeting = await direct_gemini_call(prompt, temperature=0.7, max_tokens=256)
        
        if greeting:
            logger.info(f"Generated natural greeting: {greeting}")
            return greeting
        
        logger.error("Failed to generate greeting with Gemini, using fallback")
        return "Hi! I'm Atom.ai. Let's chat about your B2B sales needs so I can find the perfect recommendations for you."
    except Exception as e:
        logger.error(f"Error generating natural greeting: {str(e)}")
        logger.error(traceback.format_exc())
        return "Hi! I'm Atom.ai. Let's chat about your B2B sales needs so I can find the perfect recommendations for you."

@app.route('/onboarding')
async def onboarding():
    """Serve the onboarding page"""
    logger.info("Serving onboarding page")
    return await render_template('onboarding.html', version=VERSION)

@app.route('/favicon.ico')
async def favicon():
    # Serve favicon from static directory
    return await serve_static('favicon.ico')

@app.route('/static/<path:filename>')
async def serve_static(filename):
    # Get the root directory of the application
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Join with the static folder
    static_folder = os.path.join(root_dir, 'static')
    # Send the requested file
    return await send_from_directory(static_folder, filename)

@app.route('/process_voice', methods=['POST'])
async def process_voice():
    """Process voice input and return transcription"""
    try:
        # Add debug logging
        logger.info("Voice processing route called")
        
        # Check if files were uploaded
        files = await request.files
        if 'audio' not in files:
            logger.error("No audio file provided in request")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = files['audio']
        logger.info(f"Received audio file: {audio_file.filename}")
        
        if audio_file.filename == '':
            logger.error("Audio file has no filename")
            return jsonify({"error": "No audio file selected"}), 400
        
        # Save the file temporarily
        temp_path = await voice_processor.save_temp_audio(audio_file)
        logger.info(f"Saved audio to temporary path: {temp_path}")
        
        # Transcribe the audio
        transcription = await voice_processor.transcribe_audio(temp_path)
        
        if not transcription:
            logger.error("Transcription failed - no text returned")
            return jsonify({"error": "Could not transcribe audio"}), 500
        
        logger.info(f"Transcribed audio: {transcription}")
        
        return jsonify({"text": transcription})
    except Exception as e:
        logger.error(f"Error processing voice: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
async def transcribe_audio():
    """API endpoint to transcribe audio using ElevenLabs"""
    try:
        # Check if files were uploaded
        files = await request.files
        if 'audio' not in files:
            return jsonify({
                "success": False,
                "error": "No audio file provided"
            }), 400
            
        audio_file = files['audio']
        
        # Save the audio file temporarily
        audio_path = await voice_processor.save_temp_audio(audio_file)
        
        # Transcribe the audio
        transcription = await voice_processor.transcribe_audio(audio_path)
        
        if not transcription:
            return jsonify({
                "success": False,
                "error": "Failed to transcribe audio"
            }), 500
            
        return jsonify({
            "success": True,
            "text": transcription
        })
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

ENABLE_TTS = True

@app.route('/api/onboarding', methods=['POST'])
async def handle_onboarding():
    """Handle onboarding conversation."""
    data = await request.get_json()
    step = data.get('step')
    answer = data.get('answer', '')
    is_follow_up = data.get('is_follow_up', False)
    follow_up_count = data.get('follow_up_count', 0)
    
    # Extract thinking process if present
    thinking_process = ""
    if "<think>" in answer and "</think>" in answer:
        start_idx = answer.find("<think>") + len("<think>")
        end_idx = answer.find("</think>")
        thinking_process = answer[start_idx:end_idx].strip()
        # Remove the thinking process from the answer
        answer = answer.replace(answer[start_idx-len("<think>"):end_idx+len("</think>")], "").strip()
        # Log the thinking process
        logger.info(f"Thinking process for step {step}: {thinking_process}")
    
    # Initialize flow controller if not already done
    global flow_controller
    if flow_controller is None:
        flow_controller = FlowController()
    
    try:
        # Update the flow controller with the current answer
        if step == 'product':
            flow_controller.current_product_line = answer
            # Generate initial keywords based on product
            try:
                flow_controller.keywords = await app.keyword_generator._generate_keywords_with_llm(f"Product: {answer}")
                logger.info(f"Generated initial keywords from product: {flow_controller.keywords}")
            except Exception as e:
                logger.error(f"Error generating initial keywords: {str(e)}")
                
        elif step == 'market':
            flow_controller.current_sector = answer
            # Update keywords based on product and market
            try:
                combined_context = f"Product: {flow_controller.current_product_line}. Market: {answer}."
                new_keywords = await app.keyword_generator._generate_keywords_with_llm(combined_context)
                # Merge and deduplicate keywords
                flow_controller.keywords = list(set(flow_controller.keywords + new_keywords))
                logger.info(f"Updated keywords with market info: {flow_controller.keywords}")
            except Exception as e:
                logger.error(f"Error updating keywords with market info: {str(e)}")
                
        elif step == 'differentiation':
            flow_controller.product_differentiation = answer
            # Update keywords based on product and differentiation
            try:
                combined_context = f"Product: {flow_controller.current_product_line}. Differentiation: {answer}."
                new_keywords = await app.keyword_generator._generate_keywords_with_llm(combined_context)
                # Merge and deduplicate keywords
                flow_controller.keywords = list(set(flow_controller.keywords + new_keywords))
                logger.info(f"Updated keywords with differentiation info: {flow_controller.keywords}")
            except Exception as e:
                logger.error(f"Error updating keywords with differentiation info: {str(e)}")
                
        elif step == 'company_size':
            flow_controller.current_segment = answer
            
            # Update keywords with company size info if not a follow-up
            if not is_follow_up:
                try:
                    # Generate keywords using the current context
                    combined_context = f"Product: {flow_controller.current_product_line}. Market: {flow_controller.current_sector}. Company Size: {answer}."
                    new_keywords = await app.keyword_generator._generate_keywords_with_llm(combined_context)
                    # Merge and deduplicate keywords
                    flow_controller.keywords = list(set(new_keywords + flow_controller.keywords))
                    logger.info(f"Updated keywords with company size info: {flow_controller.keywords}")
                except Exception as e:
                    logger.error(f"Error updating keywords with company size info: {str(e)}")
                    # Set default keywords as fallback if no keywords have been generated yet
                    if not flow_controller.keywords:
                        flow_controller.keywords = ["B2B", "Sales", "Marketing", "Technology", "Innovation"]
                        logger.info(f"Using fallback keywords: {flow_controller.keywords}")
        
        elif step == 'linkedin':
            # Store LinkedIn consent
            consent = answer.lower() in ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
            flow_controller.linkedin_consent = consent
            logger.info(f"LinkedIn consent: {consent}")
            # Clean up keywords using Gemini Flash before recommendations
            if hasattr(flow_controller, 'keywords') and flow_controller.keywords:
                try:
                    # Clean up and organize keywords
                    cleaned_keywords = await clean_keywords_with_gemini(flow_controller.keywords)
                    flow_controller.keywords = cleaned_keywords
                    logger.info(f"Cleaned keywords before recommendations: {flow_controller.keywords}")
                except Exception as e:
                    logger.error(f"Error cleaning keywords: {str(e)}")
            # Generate recommendations immediately
            try:
                # Generate company recommendations
                global company_recommender
                if company_recommender is None:
                    company_recommender = CompanyRecommender(flow_controller)
                recommendations = await company_recommender.generate_recommendations(count=5)
                
                # Format recommendations for response
                rec_text = ""
                for i, rec in enumerate(recommendations[:3]):  # Show top 3 in the response
                    rec_text += f"{rec['name']}: {rec['reason']}\n"
                
                # Log onboarding data and recommendations to CSV
                log_onboarding_and_recommendations({
                    'product': flow_controller.current_product_line,
                    'market': flow_controller.current_sector,
                    'differentiation': flow_controller.product_differentiation,
                    'company_size': flow_controller.current_segment,
                    'keywords': ', '.join(flow_controller.keywords),
                    'linkedin_consent': flow_controller.linkedin_consent
                }, recommendations)
                
                # Set response text with recommendations
                response_text = f"Top recommendations: {rec_text}"
                
            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
                response_text = "Got it. Computing your recommendations now."
            
            # Skip to recommendations step
            return 'complete', response_text
        
        elif step == 'location':
            # Skip if user says "skip" or similar
            if answer.lower() not in ['skip', 'pass', 'no', 'none', 'n/a']:
                flow_controller.zip_code = answer
                logger.info(f"Location: {answer}")
        
        # Get the next question or completion message
        if is_follow_up:
            # For follow-up questions, we stay on the same step
            next_step = step
            
            # If this is the 2nd follow-up, suggest moving to the next step
            if follow_up_count >= 1:  # 0-indexed, so 1 means 2nd follow-up
                text = await flow_controller.get_follow_up_question(step, answer, follow_up_count, suggest_next=True)
            else:
                text = await flow_controller.get_follow_up_question(step, answer, follow_up_count)
        else:
            # Move to the next step
            next_step = flow_controller.get_next_step(step)
            text = await flow_controller.get_question(next_step)
        
        # Clean the response to make it more succinct and readable
        text = _clean_llm_response(text)
        
        # Check if we've completed all steps
        completed = next_step == 'complete'
        
        # Generate audio for the response if text-to-speech is enabled
        audio_data = None
        if ENABLE_TTS:
            try:
                audio_data = await voice_processor.text_to_speech(text)
            except Exception as e:
                logger.error(f"Error generating audio: {str(e)}")
        
        return jsonify({
            "success": True,
            "text": text,
            "audio": audio_data,
            "completed": completed,
            "keywords": flow_controller.keywords
        })
    except Exception as e:
        logger.error(f"Error handling onboarding message: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/onboarding/submit', methods=['POST'])
async def submit_onboarding():
    """Handle onboarding form submission"""
    try:
        data = await request.get_json()
        
        # Log the complete data object for debugging
        logger.info(f"Complete onboarding data: {json.dumps(data, indent=2)}")
        
        # Extract form data
        product = data.get('product', '')
        market = data.get('market', '')
        differentiation = data.get('differentiation', '')
        company_size = data.get('company_size', '')
        keywords = data.get('keywords', '')
        linkedin_consent = data.get('linkedin_consent', False)
        zip_code = data.get('zip_code', '')
        additional = data.get('additional', '')
        
        logger.info(f"Received onboarding form submission: Product={product}, Market={market}, Differentiation={differentiation}, Size={company_size}, Keywords={keywords}, LinkedIn={linkedin_consent}, Zip={zip_code}")
        
        # Update flow controller with the collected information
        flow_controller.current_product_line = product
        flow_controller.current_sector = market
        flow_controller.product_differentiation = differentiation
        flow_controller.current_segment = company_size
        flow_controller.keywords = keywords
        
        # Store zip code if provided
        if zip_code:
            flow_controller.zip_code = zip_code
            logger.info(f"Added zip code: {zip_code}")
        
        # Store LinkedIn consent
        flow_controller.linkedin_consent = linkedin_consent
        logger.info(f"LinkedIn consent: {linkedin_consent}")
        
        # Store additional keywords if provided
        if additional:
            # Generate keywords using the current context
            if hasattr(flow_controller, '_generate_keywords_with_llm'):
                try:
                    flow_controller.keywords = await app.keyword_generator._generate_keywords_with_llm(additional)
                    logger.info(f"Generated keywords with LLM: {flow_controller.keywords}")
                except Exception as e:
                    logger.error(f"Error generating keywords with LLM: {str(e)}")
                    # Fall back to simple keyword extraction
                    keywords = [kw.strip() for kw in additional.split(',') if kw.strip()]
                    if keywords:
                        flow_controller.keywords = keywords
            else:
                keywords = [kw.strip() for kw in additional.split(',') if kw.strip()]
                if keywords:
                    flow_controller.keywords = keywords
        
        # Generate company recommendations
        global company_recommender
        if company_recommender is None:
            company_recommender = CompanyRecommender(flow_controller)
        recommendations = await company_recommender.generate_recommendations(count=5)
        
        # Log onboarding data and recommendations to CSV
        log_onboarding_and_recommendations({
            'product': product,
            'market': market,
            'differentiation': differentiation,
            'company_size': company_size,
            'keywords': keywords,
            'zip_code': zip_code,
            'linkedin_consent': linkedin_consent
        }, recommendations)
        
        return jsonify({
            "success": True,
            "message": "Onboarding information saved successfully",
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error in onboarding form submission: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "There was an error saving your information. Please try again."
        }), 500

@app.route('/api/store_onboarding', methods=['POST'])
async def store_onboarding():
    """
    Store onboarding answers from the user.
    """
    try:
        data = await request.get_json()
        step = data.get('step', '')
        answer = data.get('answer', '')
        
        logger.info(f"Storing onboarding answer for step '{step}': {answer}")
        
        # Initialize flow controller if not already done
        global flow_controller
        if flow_controller is None:
            flow_controller = FlowController()
        
        # Store the answer based on the step
        if step == 'product':
            flow_controller.current_product_line = answer
            logger.info(f"Set product: {answer}")
            
        elif step == 'website':
            # Store website URL
            logger.info(f"Stored website: {answer}")
            
        elif step == 'differentiation':
            # Store differentiation
            logger.info(f"Stored differentiation: {answer}")
            
        elif step == 'market':
            flow_controller.current_sector = answer
            logger.info(f"Set market sector: {answer}")
            
        elif step == 'company_size':
            flow_controller.current_segment = answer
            logger.info(f"Set company size: {answer}")
            
        elif step == 'location':
            flow_controller.zip_code = answer
            logger.info(f"Set location: {answer}")
            
        elif step == 'linkedin':
            consent = answer.lower() in ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
            flow_controller.linkedin_consent = consent
            logger.info(f"Set LinkedIn consent: {consent}")
        
        return jsonify({
            'success': True
        })
    except Exception as e:
        logger.error(f"Error in store_onboarding: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

class FlowController:
    """Controller for managing the conversation flow."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of FlowController."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the flow controller."""
        self.current_step = 'product'
        self.current_product_line = ''
        self.current_sector = ''
        self.current_segment = ''
        self.keywords = []
        self.conversation_memory = []
        
    def update_keywords(self, text):
        """Update keywords based on user input."""
        if not text:
            return  # Skip empty inputs
            
        # Initialize keywords list if it doesn't exist
        if not hasattr(self, 'keywords') or not self.keywords:
            self.keywords = []
            
        # Handle special case for acronyms and short terms
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            # Check if it's an acronym (all caps) or very short term
            if word.isupper() or (len(word) <= 5 and len(word) >= 2):
                if word and word.upper() not in [k.upper() for k in self.keywords]:
                    self.keywords.append(word.upper())
            # For normal words
            elif len(word) > 2 and word.lower() not in STOPWORDS:
                if word.capitalize() not in [k for k in self.keywords]:
                    self.keywords.append(word.capitalize())
                    
        # Extract multi-word terms
        multi_word_terms = re.findall(r'\b[A-Za-z][A-Za-z\s]{2,25}[A-Za-z]\b', text)
        for term in multi_word_terms:
            term = term.strip()
            if len(term.split()) in [2, 3] and all(len(word) > 2 for word in term.split()):
                if term.lower() not in [k.lower() for k in self.keywords]:
                    self.keywords.append(term)
        
        logger.info(f"Updated keywords from text: {self.keywords}")
    
    def get_next_step(self, current_step):
        """Get the next step in the conversation flow."""
        step_sequence = ['product', 'market', 'company_size', 'complete']
        try:
            current_index = step_sequence.index(current_step)
            if current_index < len(step_sequence) - 1:
                return step_sequence[current_index + 1]
            else:
                return 'complete'
        except ValueError:
            return 'product'  # Default to first step if current step not found

    def get_question_for_step(self, step):
        """Get the question to ask for a given step."""
        questions = {
            'product': "What product or service do you sell?",
            'market': "What market or industry do you target?",
            'company_size': "What size companies do you typically sell to? (e.g., SMB, Mid-Market, Enterprise)",
            'complete': "Thank you for providing this information! I'll use it to find relevant companies for you."
        }
        return questions.get(step, "What would you like to know?")

    def get_hint_for_step(self, step):
        """Get the hint for a given step."""
        hints = {
            'product': "e.g., CRM software, IT consulting, cloud storage",
            'market': "e.g., healthcare, finance, education",
            'company_size': "e.g., enterprise, mid-market, SMB",
            'location': "e.g., 94105 or San Francisco",
            'linkedin_consent': "Yes or No",
            'differentiation': "e.g., AI-powered, industry-specific, cost-effective"
        }
        return hints.get(step, "")

    async def get_question(self, step):
        """Async wrapper for get_question_for_step."""
        return self.get_question_for_step(step)
        
    async def generate_personalized_response(self, step, previous_answer=None):
        """Generate a personalized response for the current step using Gemini 2.0 Flash."""
        try:
            # Build context from previous answers
            context = self._build_conversation_context()
            
            # Create a prompt for Gemini based on the current step and previous answers
            prompt = self._create_step_prompt(step, previous_answer, context)
            
            # Log the prompt for debugging
            logger.info(f"Personalized response prompt: {prompt[:150]}...")
            
            # Call Gemini API
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("No Gemini API key found for personalized response")
                return self.get_question_for_step(step)
                
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.9,  # Higher temperature for more creative, casual responses
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=data,
                    timeout=15.0
                )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        personalized_response = content["parts"][0]["text"].strip()
                        logger.info(f"Generated personalized response for step {step}: {personalized_response[:100]}...")
                        return personalized_response
            
            logger.error(f"Error or unexpected response from Gemini API: {response.status_code}")
            return self.get_question_for_step(step)
            
        except Exception as e:
            logger.error(f"Error generating personalized response: {str(e)}")
            return self.get_question_for_step(step)
    
    def _build_conversation_context(self):
        """Build context from previous conversation for personalized responses."""
        context = []
        
        if hasattr(self, 'current_product_line') and self.current_product_line:
            context.append(f"Product/Service: {self.current_product_line}")
            
        if hasattr(self, 'current_sector') and self.current_sector:
            context.append(f"Target Market/Industry: {self.current_sector}")
            
        if hasattr(self, 'current_segment') and self.current_segment:
            context.append(f"Target Company Size: {self.current_segment}")
            
        if hasattr(self, 'keywords') and self.keywords:
            context.append(f"Keywords: {', '.join(self.keywords[:10])}")
            
        return "\n".join(context)
    
    def _create_step_prompt(self, step, previous_answer, context):
        """Create a prompt for Gemini based on the current step and previous answers."""
        base_prompt = f"""You are Atom, a friendly, helpful B2B sales assistant with a warm, conversational style. You're helping a sales professional with their company research.

Current conversation context:
{context}

User's most recent answer: "{previous_answer}"

Based on this context, generate a natural, conversational response for the '{step}' step of the onboarding process. Your response should:
1. Acknowledge specific details from the user's answer with genuine interest
2. Connect their answer to how it will help you find better recommendations for them
3. Provide 1-2 relevant specific examples or suggestions that could help the user
4. Ask the next question in a warm, conversational way
5. Use natural language with contractions (like "you're", "I'll", "that's")
6. Sound like a knowledgeable friend rather than a formal assistant
7. Keep your response very concise (2-3 sentences maximum)

For reference, here's what you need to include in this step:
"""

        # Add step-specific instructions and examples
        if step == 'product':
            base_prompt += """
Ask what product or service they sell, showing excitement about learning about their business.

For example, if they mention software, you might briefly suggest: 'A CRM system or data analytics tool?'

Be concise but helpful with any suggestions.
"""
        elif step == 'market':
            base_prompt += """
Ask what market or industry they target, acknowledging how their product could be valuable in different sectors.

Briefly suggest 1-2 relevant industries like: 'Healthcare, Finance, or Technology?'

Keep your suggestions relevant to their product if possible.
"""
        elif step == 'company_size':
            base_prompt += """
Ask what size companies they typically sell to, relating this to targeting strategy.

Briefly mention 1-2 relevant company size categories: 'Enterprise (1000+ employees) or Mid-market (100-999 employees)?'

Keep it short but informative.
"""
        elif step == 'differentiation':
            base_prompt += """
Ask what makes their product or service unique compared to competitors.

Briefly suggest 1-2 differentiation angles like: 'Is it your technology or perhaps your customer service?'

Keep suggestions simple and direct.
"""
        elif step == 'location':
            base_prompt += """
Ask for their location or zip code to help find relevant local events.

Briefly explain how it helps: 'This helps find nearby industry events or regional opportunities.'

Keep your explanation short and clear.
"""
        elif step == 'linkedin':
            base_prompt += """
Ask if they'd like to include LinkedIn data in their search (yes/no).

Briefly explain 1 specific benefit: 'This helps identify decision-makers at target companies.'

Keep it simple and straightforward.
"""
        else:
            base_prompt += "Ask how you can help them with their B2B research today in a friendly, conversational way."
            
        return base_prompt

    async def process_user_input(self, text, step):
        """Process user input and generate a response."""
        # Store the input based on the step
        if step == 'product':
            self.current_product_line = text
        elif step == 'market':
            self.current_sector = text
        elif step == 'company_size':
            self.current_segment = text
        
        # Update conversation memory
        if not hasattr(self, 'conversation_memory'):
            self.conversation_memory = []
        
        # Add user message to conversation memory
        self.conversation_memory.append({"role": "user", "content": text, "step": step})
        
        # Limit conversation memory to last 10 messages
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
        
        # Create conversation context from memory
        conversation_context = ""
        for i, msg in enumerate(self.conversation_memory[-5:]):  # Use last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
        
        # Generate keywords based on the updated conversation context
        if not hasattr(self, 'keywords') or not self.keywords:
            self.keywords = []
        
        try:
            new_keywords = await app.keyword_generator._generate_keywords_with_llm(conversation_context)
            # Accumulate all keywords without limiting
            self.keywords = list(set(new_keywords + self.keywords))
            logger.info(f"Updated keywords: {self.keywords}")
        except Exception as e:
            logger.error(f"Error generating keywords: {str(e)}")

class CompanyRecommender:
    """Recommends target companies based on user preferences."""
    
    def __init__(self, flow_controller):
        """Initialize the company recommender."""
        self.flow_controller = flow_controller
        # Import the real CompanyRecommender
        from company_recommender import CompanyRecommender as RealCompanyRecommender
        self.real_recommender = RealCompanyRecommender(flow_controller)
    
    async def generate_recommendations(self, count=3, verify=True):
        """Get company recommendations based on user's onboarding answers."""
        try:
            # Use the real implementation
            return await self.real_recommender.generate_recommendations(count=count, verify=verify)
        except Exception as e:
            logger.error(f"Error using real recommender: {str(e)}")
            # Fallback to mock implementation with different recommendations based on company size
            company_size = self.flow_controller.company_size if hasattr(self.flow_controller, 'company_size') else ""
            
            if company_size and "smb" in company_size.lower():
                # SMB recommendations
                recommendations = [
                    {
                        "name": "Zoho",
                        "description": "Zoho offers a suite of business, collaboration and productivity applications for small and medium businesses.",
                        "match_score": 95,
                        "reason": "Affordable analytics solutions specifically designed for small businesses"
                    },
                    {
                        "name": "Tableau",
                        "description": "Tableau helps people see and understand data with interactive visualizations.",
                        "match_score": 90,
                        "reason": "User-friendly data visualization tool popular with small businesses"
                    },
                    {
                        "name": "Domo",
                        "description": "Domo is a cloud-based business intelligence platform that helps companies make better decisions through data.",
                        "match_score": 85,
                        "reason": "Cloud-based analytics platform with SMB-friendly pricing"
                    }
                ]
            else:
                # Enterprise recommendations
                recommendations = [
                    {
                        "name": "Databricks",
                        "description": "Databricks is a data and AI company that helps organizations accelerate innovation by unifying data, analytics, and AI.",
                        "match_score": 95,
                        "reason": "Leading data platform with strong focus on AI and machine learning"
                    },
                    {
                        "name": "Snowflake",
                        "description": "Snowflake is a cloud-based data warehousing company that provides a data platform for data storage, processing, and analytic solutions.",
                        "match_score": 90,
                        "reason": "Cloud data platform competing in the same space as Databricks"
                    },
                    {
                        "name": "Confluent",
                        "description": "Confluent is a full-scale data streaming platform that enables you to easily access, store, and manage data as continuous, real-time streams.",
                        "match_score": 85,
                        "reason": "Data streaming platform that complements data processing solutions"
                    }
                ]
            
            # Limit to requested count
            return recommendations[:count]

def log_onboarding_and_recommendations(onboarding_data, recommendations=None):
    """
    Log onboarding data and recommendations to CSV for analysis
    
    Args:
        onboarding_data (dict): The onboarding data submitted by the user
        recommendations (list): The recommendations generated based on the onboarding data
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the onboarding data row
    onboarding_row = {
        'timestamp': timestamp,
        'product': onboarding_data.get('product', ''),
        'market': onboarding_data.get('market', ''),
        'differentiation': onboarding_data.get('differentiation', ''),
        'company_size': onboarding_data.get('company_size', ''),
        'keywords': ', '.join(onboarding_data.get('keywords', [])),
        'zip_code': onboarding_data.get('zip_code', ''),
        'linkedin_consent': onboarding_data.get('linkedin_consent', False)
    }
    
    # Create the onboarding CSV file if it doesn't exist
    onboarding_file = 'onboarding_data.csv'
    onboarding_file_exists = os.path.isfile(onboarding_file)
    
    with open(onboarding_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=onboarding_row.keys())
        if not onboarding_file_exists:
            writer.writeheader()
        writer.writerow(onboarding_row)
    
    logger.info(f"Logged onboarding data to {onboarding_file}")
    
    # If recommendations are provided, log them as well
    if recommendations:
        # Prepare the recommendations rows
        recommendations_rows = []
        events_rows = []
        
        for i, rec in enumerate(recommendations):
            # Log main recommendation data
            rec_row = {
                'timestamp': timestamp,
                'recommendation_index': i + 1,
                'company_name': rec.get('name', ''),
                'description': rec.get('description', '')[:200],  # Truncate long descriptions
                'fit_reason': rec.get('fit_reason', '')[:200],  # Truncate long fit reasons
                'fit_score': rec.get('fit_score', {}).get('overall_score', 0),
                'product_fit': rec.get('fit_score', {}).get('product_fit', 0),
                'market_fit': rec.get('fit_score', {}).get('market_fit', 0),
                'size_fit': rec.get('fit_score', {}).get('size_fit', 0),
                'keyword_fit': rec.get('fit_score', {}).get('keyword_fit', 0),
                'website': rec.get('website', ''),
                'key_personnel_count': len(rec.get('key_personnel', [])),
                'news_count': len(rec.get('recent_news', [])),
                'events_count': len(rec.get('events', [])),
                'product': onboarding_data.get('product', ''),
                'market': onboarding_data.get('market', ''),
                'company_size': onboarding_data.get('company_size', '')
            }
            recommendations_rows.append(rec_row)
            
            # Log events data separately
            for j, event in enumerate(rec.get('events', [])):
                event_row = {
                    'timestamp': timestamp,
                    'recommendation_index': i + 1,
                    'company_name': rec.get('name', ''),
                    'event_index': j + 1,
                    'event_name': event.get('name', ''),
                    'event_date': event.get('date', ''),
                    'event_location': event.get('location', ''),
                    'event_url': event.get('url', ''),
                    'event_description': event.get('description', '')[:200],  # Truncate long descriptions
                    'product': onboarding_data.get('product', ''),
                    'market': onboarding_data.get('market', ''),
                    'company_size': onboarding_data.get('company_size', '')
                }
                events_rows.append(event_row)
        
        # Create the recommendations CSV file if it doesn't exist
        recommendations_file = 'recommendations_data.csv'
        recommendations_file_exists = os.path.isfile(recommendations_file)
        
        with open(recommendations_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=recommendations_rows[0].keys())
            if not recommendations_file_exists:
                writer.writeheader()
            writer.writerows(recommendations_rows)
        
        logger.info(f"Logged {len(recommendations_rows)} recommendations to {recommendations_file}")
        
        # Create the events CSV file if it doesn't exist
        if events_rows:
            events_file = 'events_data.csv'
            events_file_exists = os.path.isfile(events_file)
            
            with open(events_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=events_rows[0].keys())
                if not events_file_exists:
                    writer.writeheader()
                writer.writerows(events_rows)
            
            logger.info(f"Logged {len(events_rows)} events to {events_file}")

@app.route('/recommendations-view')
async def recommendations_view_page():
    session = await request.scope.get("session", {})
    recommendations = session.get("latest_recommendations", {})
    return await render_template("recommendations.html", recommendations=recommendations)

@app.route('/new-recommendations')
async def new_recommendations_page():
    logger.info("Serving new recommendations page")
    return await render_template("recommendations.html", version=VERSION, is_new=True)

@app.route('/recommendations')
async def recommendations():
    """Serve the recommendations page"""
    logger.info("Serving recommendations page")
    return await render_template('recommendations.html', version=VERSION)

@app.route('/api/recommendations', methods=['GET', 'POST'])
async def get_recommendations():
    """
    Get company recommendations based on user's onboarding answers
    """
    try:
        # Get recommendations from company recommender
        global company_recommender, flow_controller
        if company_recommender is None and flow_controller is not None:
            company_recommender = CompanyRecommender(flow_controller)
        elif company_recommender is None:
            # Initialize flow controller if needed
            flow_controller = FlowController()
            company_recommender = CompanyRecommender(flow_controller)
        
        # Check if verification is requested (default to true)
        verify = request.args.get('verify', 'true').lower() == 'true'
        logger.info(f"Generating recommendations with verification={verify}")
        
        # Generate recommendations with verification
        recommendations = await company_recommender.generate_recommendations(count=5, verify=verify)
        
        # Log the recommendations if we have onboarding data
        if hasattr(flow_controller, 'current_product_line') and flow_controller.current_product_line:
            onboarding_data = {
                'product': flow_controller.current_product_line,
                'market': flow_controller.current_sector if hasattr(flow_controller, 'current_sector') else '',
                'differentiation': flow_controller.product_differentiation if hasattr(flow_controller, 'product_differentiation') else '',
                'company_size': flow_controller.current_segment if hasattr(flow_controller, 'current_segment') else '',
                'keywords': flow_controller.keywords if hasattr(flow_controller, 'keywords') else []
            }
            log_onboarding_and_recommendations(onboarding_data, recommendations)
        
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/generate_keywords', methods=['POST'])
async def generate_keywords():
    """Generate keywords based on the provided context"""
    try:
        data = await request.get_json()
        context = data.get('context', {})
        
        # Extract context information
        product = context.get('product', '')
        market = context.get('market', '')
        company_size = context.get('company_size', '')
        additional_context = context.get('additional_context', '')
        
        # Combine context information
        combined_context = f"Product: {product}. Market: {market}. Company Size: {company_size}. Additional Context: {additional_context}"
        
        logger.info(f"Generating keywords for context: {combined_context}")
        
        # Force LLM usage, bypass any mock data settings
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("No Gemini API key found. Cannot generate keywords.")
            return jsonify({
                "success": False,
                "error": "No Gemini API key found"
            }), 500
        
        # Prepare the prompt for keyword generation
        prompt = f"""
        As a research assistant, generate as many relevant keywords as possible for a company that:
        - Offers: {product}
        - Targets: {market}
        - Focuses on {company_size} companies
        - Additional context: {additional_context}
        
        Generate only the most relevant keywords that would help find ideal target companies.
        Format your response as a comma-separated list of keywords only, without any additional text or explanations.
        """
        
        # Make the API request directly
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=data,
                timeout=10.0  # Reduced timeout for better performance
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Gemini API: {response.status_code}, {response.text}")
                return jsonify({
                    "success": False,
                    "error": f"Gemini API error: {response.status_code}"
                }), 500
            
            response_data = response.json()
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                content = response_data["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    keywords_text = content["parts"][0]["text"]
                    
                    # Parse the keywords from the response
                    keywords = [kw.strip() for kw in keywords_text.split(',')]
                    
                    # Limit to 20 keywords
                    keywords = keywords[:20]
                else:
                    logger.error(f"Unexpected response format from Gemini API: {response_data}")
                    return jsonify({
                        "success": False,
                        "error": "Failed to parse keywords from Gemini API response"
                    }), 500
            else:
                logger.error(f"Unexpected response format from Gemini API: {response_data}")
                return jsonify({
                    "success": False,
                    "error": "Failed to parse keywords from Gemini API response"
                }), 500
        
        # Store keywords in flow controller
        flow_controller.keywords = keywords
        
        return jsonify({
            "success": True,
            "keywords": keywords
        })
    except Exception as e:
        logger.error(f"Error generating keywords: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/onboarding/question')
async def get_onboarding_question():
    """API endpoint to get the question for a specific onboarding step"""
    step = request.args.get('step', 'product')
    previous_answer = request.args.get('previous_answer', '')
    logger.info(f"Getting question for step: {step}")
    
    # Initialize flow controller if not already done
    global flow_controller
    if flow_controller is None:
        flow_controller = FlowController()
    
    # Get the question for the current step using Gemini 2.0 Flash
    try:
        # Generate personalized question using Gemini
        question = await flow_controller.generate_personalized_response(step, previous_answer)
        logger.info(f"Generated personalized question for {step}: {question[:100]}...")
        
        # If Gemini fails, fall back to default question
        if not question or "error" in question.lower():
            logger.warning(f"Using fallback question for {step}")
            question = flow_controller.get_question_for_step(step)
    except Exception as e:
        logger.error(f"Error generating personalized question: {str(e)}")
        # Fall back to default question
        question = flow_controller.get_question_for_step(step)
    
    # Generate audio for the question
    audio_base64 = None
    try:
        if settings.get('tts_enabled', True):
            # Initialize voice processor if needed
            global voice_processor
            if voice_processor is None:
                voice_processor = VoiceProcessor()
            audio_base64 = await voice_processor.text_to_speech(question)
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
    
    # Return the question, hint, and any audio
    return jsonify({
        'success': True,
        'text': question,
        'hint': flow_controller.get_hint_for_step(step),
        'audio': audio_base64,
        'keywords': flow_controller.keywords if hasattr(flow_controller, 'keywords') and flow_controller.keywords else []
    })

@app.route('/api/onboarding/followup', methods=['POST'])
async def onboarding_followup():
    """API endpoint to handle follow-up questions during onboarding using Gemini Flash 2.0"""
    try:
        # Get the request data
        data = await request.get_json()
        step = data.get('step')
        question = data.get('question')
        
        if not step or not question:
            return jsonify({
                "success": False,
                "error": "Missing required parameters"
            }), 400
            
        logger.info(f"Follow-up question for step {step}: {question}")
        
        # Define context for different onboarding steps
        step_contexts = {
            'product': "This is about the product or service the user sells. It might include questions about product categories, features, or technical details.",
            'market': "This is about market segments like Enterprise (large companies with 1000+ employees), SMB (small and medium businesses with fewer than 1000 employees), or Consumer (individual customers).",
            'differentiation': "This is about what makes the user's product or service unique compared to others in the market.",
            'company_size': "This is about the size of the company where the user works, such as startup, small business, mid-market, or enterprise.",
            'linkedin': "This is about whether the user consents to LinkedIn data integration.",
            'location': "This is about the user's geographic area of focus.",
            'recommendations': "This is about the user's preferences for company recommendations."
        }
        
        # Get the context for the current step
        context = step_contexts.get(step, "This is part of the onboarding process for a research assistant.")
        
        # Prepare the prompt for Gemini
        prompt = f"""You are a helpful assistant guiding a user through an onboarding process for a research assistant.
        
Current onboarding step: {step}
Context: {context}

The user has asked the following question:
"{question}"

Please provide a clear, concise, and helpful response that directly answers their question.
Keep your response under 100 words and focus only on providing the information requested.
"""
        
        # Initialize voice processor for TTS
        global voice_processor
        if voice_processor is None:
            voice_processor = VoiceProcessor()
        
        # Call Gemini Flash 2.0 API
        from question_engine import QuestionEngine
        question_engine = QuestionEngine()
        response_text = await question_engine.get_gemini_response(prompt)
        
        if not response_text:
            return jsonify({
                "success": False,
                "error": "Failed to generate response"
            }), 500
            
        # Generate audio if TTS is enabled
        audio_b64 = None
        if ENABLE_TTS:
            try:
                audio_b64 = await voice_processor.text_to_speech(response_text)
            except Exception as e:
                logger.error(f"Error generating speech: {str(e)}")
                # Continue without audio if TTS fails
        
        return jsonify({
            "success": True,
            "text": response_text,
            "audio": audio_b64
        })
    except Exception as e:
        logger.error(f"Error processing follow-up question: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/data-logs')
async def data_logs_page():
    """Serve the data logs page"""
    logger.info("Serving data logs page")
    
    # Read the CSV files
    onboarding_data = []
    recommendations_data = []
    
    try:
        # Read onboarding data
        if os.path.exists('onboarding_data.csv'):
            with open('onboarding_data.csv', 'r') as f:
                reader = csv.DictReader(f)
                onboarding_data = list(reader)
        
        # Read recommendations data
        if os.path.exists('recommendations_data.csv'):
            with open('recommendations_data.csv', 'r') as f:
                reader = csv.DictReader(f)
                recommendations_data = list(reader)
    except Exception as e:
        logger.error(f"Error reading data logs: {str(e)}")
    
    return await render_template('data_logs.html', 
                                onboarding_data=onboarding_data, 
                                recommendations_data=recommendations_data,
                                version=VERSION)

@app.route('/api/ws')
async def ws():
    """Handle WebSocket connections"""
    logger.info("New WebSocket connection established")
    try:
        # Send initial greeting
        greeting = "Which company do you work for?"
        greeting_audio = await voice_processor.text_to_speech(greeting)
        if greeting_audio:
            await websocket.send_json({
                "success": True,
                "response": greeting,
                "audio": greeting_audio,
                "step": "company"
            })
        
        while True:
            # Receive audio data
            audio_data = await websocket.receive()
            if not audio_data:
                continue
                
            # Process voice command
            result = await voice_processor.process_voice_command(audio_data)
            
            # Format response
            response = {
                "success": "error" not in result,
                "response": result.get("response", ""),
                "audio": result.get("audio", ""),
                "step": result.get("step", "company"),
                "error": result.get("error", None)
            }
            
            # Send response
            await websocket.send_json(response)
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "success": False,
            "error": str(e)
        })

@app.route('/onboarding_data.csv')
async def serve_onboarding_data():
    """Serve the onboarding data CSV file for download"""
    logger.info("Serving onboarding data CSV file")
    
    if not os.path.exists('onboarding_data.csv'):
        return "File not found", 404
    
    return await send_file('onboarding_data.csv', 
                          mimetype='text/csv',
                          as_attachment=True,
                          attachment_filename='onboarding_data.csv')

@app.route('/recommendations_data.csv')
async def serve_recommendations_data():
    """Serve the recommendations data CSV file for download"""
    logger.info("Serving recommendations data CSV file")
    
    if not os.path.exists('recommendations_data.csv'):
        return "File not found", 404
    
    return await send_file('recommendations_data.csv', 
                          mimetype='text/csv',
                          as_attachment=True,
                          attachment_filename='recommendations_data.csv')

@app.route('/events_data.csv')
async def serve_events_data():
    """Serve the events data CSV file for download"""
    try:
        # Set the appropriate headers for CSV download
        response = await make_response(await send_file('events_data.csv'))
        response.headers['Content-Disposition'] = 'attachment; filename=events_data.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        # Add headers to prevent caching
        response = await add_header(response)
        
        return response
    except Exception as e:
        logger.error(f"Error serving events data CSV: {str(e)}")
        return await make_response("Error serving events data CSV", 500)

@app.route('/api/save_interaction', methods=['POST'])
async def save_interaction():
    """
    Save user interaction to CSV for journey tracking
    """
    try:
        data = await request.get_json()
        
        # Validate required fields
        if not all(key in data for key in ['timestamp', 'userInput']):
            return jsonify({
                "success": False,
                "error": "Missing required fields"
            }), 400
            
        # Ensure the directory exists
        os.makedirs('data/user_journey', exist_ok=True)
        
        # Create or append to CSV file
        csv_path = 'data/user_journey/user_interactions.csv'
        file_exists = os.path.isfile(csv_path)
        
        # Determine all possible fields from the data
        all_fields = ['timestamp', 'userInput', 'keywords', 'question', 'product', 'market', 
                     'company_size', 'assistant_response', 'question_type']
        
        # Filter to only include fields that exist in the data
        fieldnames = [field for field in all_fields if field in data]
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Create a row with only the fields that exist in the data
            row = {field: data[field] for field in fieldnames}
            writer.writerow(row)
            
        # Update the flow controller with this interaction data
        global flow_controller
        if flow_controller is None:
            flow_controller = FlowController()
        
        # Update product, market, and company size if provided
        if 'product' in data and data['product']:
            flow_controller.current_product_line = data['product']
        if 'market' in data and data['market']:
            flow_controller.current_sector = data['market']
        if 'company_size' in data and data['company_size']:
            flow_controller.current_segment = data['company_size']
            
        # Update keywords if provided
        if 'keywords' in data and data['keywords']:
            # Convert comma-separated string to list if needed
            if isinstance(data['keywords'], str):
                new_keywords = [k.strip() for k in data['keywords'].split(',') if k.strip()]
            else:
                new_keywords = data['keywords']
                
            # Merge with existing keywords
            flow_controller.keywords = list(set(flow_controller.keywords + new_keywords))
            # Cap at 20 keywords
            flow_controller.keywords = flow_controller.keywords[:20]
            
        logger.info(f"Saved user interaction: {data['userInput'][:50]}...")
        logger.info(f"Updated flow controller keywords: {flow_controller.keywords}")
            
        return jsonify({
            "success": True,
            "message": "User interaction saved successfully"
        })
        
    except Exception as e:
        logger.error(f"Error saving user interaction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/user_journey')
async def user_journey():
    """Serve the user journey page"""
    try:
        # Check if the user journey CSV exists
        csv_path = 'data/user_journey/user_interactions.csv'
        journey_data = []
        
        if os.path.isfile(csv_path):
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    journey_data.append(row)
        
        return await render_template('user_journey.html', 
                                    version=VERSION, 
                                    journey_data=journey_data)
    except Exception as e:
        logger.error(f"Error serving user journey page: {str(e)}")
        return await render_template('user_journey.html', 
                                    version=VERSION, 
                                    journey_data=[],
                                    error=str(e))

@app.after_request
async def add_header(response):
    """Add headers to prevent caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/api/feedback', methods=['POST'])
async def store_feedback():
    """API endpoint to store user feedback about recommendations"""
    try:
        data = await request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Required fields
        company_name = data.get('company_name')
        feedback_text = data.get('feedback')
        
        if not company_name or not feedback_text:
            return jsonify({"error": "Missing required fields: company_name, feedback"}), 400
        
        # Get user ID from session or use default
        user_id = session.get('user_id', 'default_user')
        
        # Initialize user memory
        user_memory = UserMemory(user_id)
        
        # Extract preferences from feedback text
        preferences = user_memory.extract_preferences_from_feedback(feedback_text)
        
        # Store company preference
        if preferences.get('preference'):
            user_memory.store_company_preference(
                company_name,
                preferences['preference'],
                preferences.get('reason')
            )
        
        # Store recommendation feedback if full recommendation data is provided
        recommendation_data = data.get('recommendation_data')
        if recommendation_data:
            user_memory.store_recommendation_feedback(
                recommendation_data,
                {
                    'feedback_text': feedback_text,
                    'preference': preferences.get('preference'),
                    'reason': preferences.get('reason')
                }
            )
        
        logger.info(f"Stored feedback for company {company_name}: {preferences.get('preference')}")
        
        return jsonify({
            "success": True,
            "message": "Feedback stored successfully",
            "preference_detected": preferences.get('preference')
        })
    
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/preferences', methods=['GET'])
async def get_user_preferences():
    """API endpoint to get user preferences"""
    try:
        # Get user ID from session or use default
        user_id = session.get('user_id', 'default_user')
        
        # Initialize user memory
        user_memory = UserMemory(user_id)
        
        # Get memory summary
        preferences = user_memory.get_memory_summary()
        
        return jsonify(preferences)
    
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice_interaction', methods=['POST'])
async def voice_interaction():
    """API endpoint to process voice interactions"""
    try:
        # Get the request data
        data = await request.get_json()
        text = data.get('text', '')
        step = data.get('step', 'product')
        
        logger.info(f"Voice interaction received: {text} (step: {step})")
        
        # Initialize flow controller if needed
        global flow_controller
        if flow_controller is None:
            flow_controller = FlowController()
        
        # Initialize voice processor if needed
        global voice_processor
        if voice_processor is None:
            voice_processor = VoiceProcessor()
        
        # Process the user input based on the current step
        next_step = step
        
        # Process based on current step
        if step == 'product':
            # First step - set the company name
            flow_controller.company_name = text
            next_step = 'market'
            
            # Generate keywords for the company
            try:
                keywords = await app.keyword_generator._generate_keywords_with_llm(text)
                flow_controller.keywords = keywords
                logger.info(f"Generated keywords: {keywords}")
            except Exception as e:
                logger.error(f"Error generating keywords: {str(e)}")
                flow_controller.keywords = ["sales", "marketing", "business"]
            
        elif step == 'market':
            flow_controller.market = text
            next_step = 'differentiation'
            
        elif step == 'differentiation':
            flow_controller.differentiation = text
            next_step = 'company_size'
            
        elif step == 'company_size':
            flow_controller.company_size = text
            next_step = 'zip_code'
            
        elif step == 'zip_code':
            flow_controller.zip_code = text
            next_step = 'linkedin'
            
        elif step == 'linkedin':
            flow_controller.linkedin_consent = text.lower() in ['yes', 'y', 'sure', 'ok', 'okay', 'yep', 'yeah']
            next_step = 'complete'
            
            # Generate recommendations immediately
            recommendations = []
            try:
                # Use the company recommender to generate recommendations
                global company_recommender
                if company_recommender is None:
                    company_recommender = CompanyRecommender(flow_controller)
                recommendations = await company_recommender.generate_recommendations(count=5)
                logger.info(f"Generated recommendations: {recommendations}")
            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
            
            # Create response with recommendations
            response_text = await flow_controller.generate_personalized_response('complete', text)
            if not response_text or "error" in response_text.lower():
                response_text = "Great! Based on your inputs, I've prepared detailed recommendations for you. Check out the recommendations tab to see target companies, investment areas, recent articles, key decision makers, and upcoming events."
            
            # Generate keywords if needed
            keywords = []
            if hasattr(flow_controller, 'keywords'):
                keywords = flow_controller.keywords
                
            # Generate audio for the response
            audio_base64 = None
            try:
                if settings.get('tts_enabled', True):
                    audio_base64 = await voice_processor.text_to_speech(response_text)
            except Exception as e:
                logger.error(f"Error generating speech: {str(e)}")
            
            # Return recommendations as JSON for the frontend to display
            return jsonify({
                'success': True,
                'text': response_text,
                'audio': audio_base64,
                'next_step': next_step,
                'keywords': keywords,
                'recommendations': recommendations,
                'show_recommendations_tab': True
            })
        
        # For all other steps, generate a personalized response using Gemini
        try:
            # Update keywords from user input
            flow_controller.update_keywords(text)
            
            # Generate personalized response for the next step
            response_text = await flow_controller.generate_personalized_response(next_step, text)
            logger.info(f"Generated personalized response for {next_step}: {response_text[:100]}...")
            
            # If Gemini fails or returns an error, fall back to default questions
            if not response_text or "error" in response_text.lower():
                logger.warning(f"Using fallback response for {next_step}")
                response_text = flow_controller.get_question_for_step(next_step)
        except Exception as e:
            logger.error(f"Error generating personalized response: {str(e)}")
            response_text = flow_controller.get_question_for_step(next_step)
        
        # Generate audio for the response
        audio_base64 = None
        if settings.get('tts_enabled', True):
            try:
                audio_base64 = await voice_processor.text_to_speech(response_text)
            except Exception as e:
                logger.error(f"Error generating speech: {str(e)}")
        
        # Get current keywords to return to frontend
        keywords = []
        if hasattr(flow_controller, 'keywords'):
            keywords = flow_controller.keywords
        
        # Return the response
        return jsonify({
            'success': True,
            'text': response_text,
            'next_step': next_step,
            'audio': audio_base64,
            'keywords': keywords
        })
    except Exception as e:
        logger.error(f"Error processing voice interaction: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback: {traceback_str}")
        return jsonify({
            'success': False,
            'error': str(e),
            'text': "I'm sorry, I encountered an error processing your voice input. Please try again."
        }), 500

@app.route('/api/text_to_speech', methods=['POST'])
async def text_to_speech():
    """
    Convert text to speech using ElevenLabs TTS.
    """
    try:
        data = await request.get_json()
        text = data.get('text', '')
        
        app.logger.info(f"Text-to-speech request received: {text[:50]}...")
        
        if not text:
            app.logger.warning("No text provided for TTS")
            return jsonify({
                'error': 'No text provided'
            }), 400
            
        logger.info("Calling voice processor for TTS generation")
        global voice_processor
        if voice_processor is None:
            voice_processor = VoiceProcessor()
        audio_data = await voice_processor.text_to_speech(text)
        
        if audio_data == "QUOTA_EXCEEDED":
            # Return a special status to inform the client about quota issues
            return jsonify({
                'status': 'quota_exceeded',
                'message': 'ElevenLabs quota exceeded. Audio functionality temporarily unavailable.'
            }), 200
        elif audio_data:
            # Audio data is already base64 encoded from the text_to_speech method
            logger.info(f"Successfully generated audio, size: {len(audio_data)} bytes")
            
            return jsonify({
                'audio': audio_data
            }), 200
        else:
            logger.error("Failed to generate speech - no audio returned from voice processor")
            return jsonify({'error': 'Failed to generate speech'}), 200
    except Exception as e:
        app.logger.error(f"Error in text_to_speech: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/get_next_question', methods=['POST'])
async def get_next_question():
    """API endpoint to get the next question in the onboarding flow"""
    try:
        # Get the current step from the request
        data = await request.get_json()
        current_step = data.get('current_step')
        user_answers = data.get('user_answers', {})
        
        logger.info(f"Getting next question for step: {current_step}")
        
        # Initialize flow controller if not already done
        global flow_controller
        if flow_controller is None:
            flow_controller = FlowController()
        
        # Process the answer and determine the next step
        next_step = flow_controller.get_next_step(current_step)
        text = await flow_controller.get_question(next_step)
        
        # Get the question for the next step
        question = text
        
        # Generate audio for the question
        audio_base64 = None
        try:
            if settings.get('tts_enabled', True):
                audio_base64 = await generate_tts(question)
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
        
        # Return the question, hint, and any audio
        return jsonify({
            'success': True,
            'question': question,
            'hint': flow_controller.get_hint_for_step(next_step),
            'next_step': next_step,
            'audio': audio_base64
        })
    except Exception as e:
        logger.error(f"Error getting next question: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.after_request
async def add_header(response):
    """Add headers to prevent caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/api/process_recommendations', methods=['POST'])
async def process_recommendations():
    """API endpoint to process recommendations after onboarding"""
    try:
        # Generate company recommendations
        global company_recommender
        if company_recommender is None:
            company_recommender = CompanyRecommender(flow_controller)
        recommendations = await company_recommender.generate_recommendations(count=5)
        
        # Format recommendations for response
        rec_text = "Based on your information, here are some recommended companies:\n\n"
        for i, rec in enumerate(recommendations):
            rec_text += f"{i+1}. {rec.get('name', 'Unknown Company')}\n"
            rec_text += f"   Match: {rec.get('match_score', 0)}% - {rec.get('reason', '')}\n\n"
        
        # Log onboarding data and recommendations to CSV
        log_onboarding_and_recommendations({
            'product': flow_controller.product,
            'website': flow_controller.website,
            'differentiation': flow_controller.differentiation,
            'market': flow_controller.market,
            'company_size': flow_controller.company_size,
            'location': flow_controller.location,
            'linkedin_consent': flow_controller.linkedin_consent
        }, recommendations)
        
        return jsonify({
            "success": True,
            "text": rec_text,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

async def process_user_input(step, answer):
    """Process user input and generate a response."""
    global flow_controller
    
    if flow_controller is None:
        flow_controller = FlowController()
    
    logger.info(f"Processing user input for step {step}: {answer}")
    
    # Initialize response
    response = ""
    next_step = step
    
    try:
        # Update flow controller with the answer
        if step == 'product':
            flow_controller.current_product_line = answer
        elif step == 'market':
            flow_controller.current_sector = answer
        elif step == 'differentiation':
            flow_controller.product_differentiation = answer
        elif step == 'company_size':
            flow_controller.current_segment = answer
        elif step == 'location':
            flow_controller.zip_code = answer
        elif step == 'linkedin':
            flow_controller.linkedin_consent = answer.lower() in ['yes', 'y', 'sure', 'ok', 'okay', 'yep', 'yeah']
        
        # Update keywords based on the answer
        flow_controller.update_keywords(answer)
        
        # Determine the next step in the conversation flow
        next_step = flow_controller.get_next_step(current_step=step)
        
        # Generate a personalized response using Gemini Flash 2.0
        try:
            # Generate personalized response for the next step
            response = await flow_controller.generate_personalized_response(next_step, answer)
            logger.info(f"Generated personalized response for {next_step}: {response[:100]}...")
            
            # If Gemini fails or returns an error, fall back to simpler prompt
            if not response or "error" in response.lower():
                logger.warning(f"Using simplified prompt for {next_step}")
                
                # Create a much simpler prompt for Gemini
                product = flow_controller.current_product_line if hasattr(flow_controller, 'current_product_line') else ""
                market = flow_controller.current_sector if hasattr(flow_controller, 'current_sector') else ""
                
                # Very simple prompt
                prompt = f"""You are a friendly B2B sales assistant. 
                
The user just told you: "{answer}"

User is selling: {product}
User's target market: {market}

Give a short, friendly response that:
1. Acknowledges what they said
2. Asks about {next_step} (like what market they target or what size companies they sell to)

Keep it very conversational, brief, and casual. Don't explain what you'll do with the information.
"""
                
                # Call the simplified Gemini API function
                logger.info("Calling Gemini with simplified prompt")
                response = await direct_gemini_call(prompt, temperature=0.7)
                
                if not response:
                    # Use the default fallback if Gemini fails
                    response = f"Thanks for sharing that! What {next_step} are you targeting?"
                    logger.warning(f"Using fallback response: {response}")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            response = flow_controller.get_question_for_step(next_step)
            
        # Handle final steps and recommendations
        if next_step == 'complete' or step == 'linkedin':
            logger.info("Reached complete step, generating recommendations")
            # Generate recommendations
            try:
                # Use the company recommender to generate recommendations
                global company_recommender
                if company_recommender is None:
                    company_recommender = CompanyRecommender(flow_controller)
                recommendations = await company_recommender.generate_recommendations(count=5)
                
                # Create a personalized completion response with Gemini
                completion_prompt = f"""You are Atom, a friendly B2B sales assistant. The user has just finished telling you about their sales needs:

Product/Service: {flow_controller.current_product_line}
Target Market: {flow_controller.current_sector}
Company Size: {flow_controller.current_segment}

Generate an enthusiastic, friendly response (2-3 sentences) that:
1. Thanks them for sharing their information
2. Tells them you've prepared personalized company recommendations based on their needs
3. Encourages them to check out the recommendations tab
4. Sounds conversational and natural, not robotic

Keep it brief but warm and helpful.
"""
                
                completion_response = await direct_gemini_call(completion_prompt, temperature=0.7)
                if completion_response:
                    response = completion_response
                else:
                    # Fallback if Gemini fails
                    response = "Thanks for sharing! I've got some great recommendations ready for you - check out the recommendations tab to see companies, articles, and events that match your needs."
                
                # Set flag to show recommendations button
                show_recommendations_button = True
            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
                response = "Thanks for the info! Your recommendations are ready in the recommendations tab."
                
    except Exception as e:
        logger.error(f"Error in process_user_input: {str(e)}")
        logger.error(traceback.format_exc())
        response = "I encountered an issue processing your input. Let's try again. What product or service are you interested in?"
        next_step = 'product'
    
    return jsonify({
        'success': True,
        'text': response,
        'audio': None,
        'next_step': next_step,
        'show_recommendations_button': show_recommendations_button if 'show_recommendations_button' in locals() else False
    })

@app.route('/api/onboarding/step', methods=['POST'])
async def onboarding_step():
    """API endpoint to get the next onboarding step"""
    try:
        # Get the request data
        data = await request.get_json()
        step = data.get('step')
        answer = data.get('answer', '')
        
        logger.info(f"Onboarding step request: step='{step}', answer='{answer}'")
        
        # Initialize flow controller if not already done
        global flow_controller
        if flow_controller is None:
            flow_controller = FlowController()
        
        # Process the answer and determine the next step
        response_data = await process_user_input(step, answer)
        
        # Extract data from the response
        next_step = response_data.get('next_step')
        response_text = response_data.get('text')
        show_recommendations_button = response_data.get('show_recommendations_button', False)
        
        # Ensure the response is meaningful but concise by checking for specific suggestions
        # If the response doesn't contain specific suggestions, enhance it with just 1-2 examples
        if next_step == 'market' and not any(sector in response_text.lower() for sector in ['healthcare', 'finance', 'retail', 'technology', 'education', 'manufacturing']):
            # Add just 1-2 industry suggestions if they're missing
            response_text += " Are you targeting sectors like Healthcare or Technology?"
        
        elif next_step == 'company_size' and not any(size in response_text.lower() for size in ['enterprise', 'mid-market', 'small', 'medium', 'smb', 'startup']):
            # Add just 1-2 company size suggestions if they're missing
            response_text += " Do you focus on Enterprise or SMB customers?"
        
        # Log the response for debugging
        logger.info(f"Generated response: {response_text[:100]}...")
        
        # Generate audio for the response
        audio_base64 = None
        try:
            if settings.get('tts_enabled', True):
                # Initialize voice processor if needed
                global voice_processor
                if voice_processor is None:
                    voice_processor = VoiceProcessor()
                audio_base64 = await voice_processor.text_to_speech(response_text)
                logger.info("Generated audio for response")
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
        
        # Return the response
        return jsonify({
            'success': True,
            'text': response_text,
            'hint': flow_controller.get_hint_for_step(next_step) if next_step else "",
            'audio': audio_base64,
            'next_step': next_step,
            'keywords': flow_controller.keywords if hasattr(flow_controller, 'keywords') and flow_controller.keywords else [],
            'show_recommendations_button': show_recommendations_button
        })
    except Exception as e:
        logger.error(f"Error processing onboarding step: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

async def clean_keywords_with_gemini(keywords):
    """Clean and organize keywords using Gemini Flash 2.0."""
    if not keywords:
        return []
    
    # Join keywords into a comma-separated string
    keywords_str = ", ".join(keywords)
    
    # Create prompt for Gemini
    prompt = f"""
    I have collected the following keywords during a conversation with a professional:
    
    {keywords_str}
    
    Please clean up and organize these keywords:
    1. Remove duplicates and similar concepts
    2. Standardize terminology
    3. Group related concepts
    4. Prioritize the most relevant terms
    5. Return only the top 30 most relevant keywords
    
    Format your response as a comma-separated list of keywords only, without any additional text or explanations.
    """
    
    try:
        # Call Gemini Flash for quick processing
        gemini_client = genai.GenerativeModel('gemini-2.0-flash')
        response = await gemini_client.generate_content_async(prompt)
        cleaned_text = response.text.strip()
        
        # Split by commas and clean up each keyword
        cleaned_keywords = [k.strip() for k in cleaned_text.split(',')]
        
        # Remove any empty strings
        cleaned_keywords = [k for k in cleaned_keywords if k]
        
        return cleaned_keywords
    except Exception as e:
        logger.error(f"Error cleaning keywords with Gemini: {str(e)}")
        # Return original keywords if cleaning fails
        return keywords

@app.route('/test_keywords', methods=['POST'])
async def test_keywords():
    """Test endpoint for keyword generation."""
    try:
        data = await request.get_json()
        text = data.get('text', '')
        
        # Create a temporary flow controller
        fc = FlowController()
        
        # Update keywords
        fc.update_keywords(text)
        
        # Log the keywords
        logger.info(f"Generated keywords from text: {fc.keywords}")
        
        return jsonify({
            'success': True,
            'keywords': fc.keywords
        })
    except Exception as e:
        logger.error(f"Error in test_keywords: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/generate_recommendations", methods=["POST"])
async def generate_recommendations():
    data = await request.get_json()
    prompt = data.get("prompt", "")
    
    response = await generate_response(prompt)
    
    # Parse the response to extract structured recommendations data
    recommendations = {
        "companies": [],
        "article_links": [],
        "quotes": [],
        "leads": [],
        "events": []
    }
    
    # Extract companies
    if "companies" in response.lower() or "company" in response.lower() or "recommended" in response.lower():
        company_matches = re.findall(r'\d+\.\s+([\w\s&\.,]+?)(?=\s*\d+\.|$|\n)', response)
        if company_matches:
            recommendations["companies"] = [company.strip() for company in company_matches if company.strip()]
        else:
            # Alternative pattern for different formatting
            company_section = re.search(r'companies[:\s]+(.*?)(?=\n\n|\Z)', response.lower(), re.DOTALL)
            if company_section:
                companies_text = company_section.group(1)
                company_candidates = re.findall(r'[\w\s&\.,]+', companies_text)
                recommendations["companies"] = [c.strip() for c in company_candidates if c.strip() and len(c.strip()) > 3]
    
    # If no companies found yet, try simple extraction
    if not recommendations["companies"]:
        # Simple extraction based on common patterns
        lines = response.split('\n')
        for line in lines:
            if re.search(r'^\d+\.\s+', line):  # Numbered list
                company = re.sub(r'^\d+\.\s+', '', line).strip()
                if company and len(company) > 3:
                    recommendations["companies"].append(company)
    
    # Extract article links
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    article_links = re.findall(url_pattern, response)
    recommendations["article_links"] = article_links
    
    # Extract quotes
    quote_patterns = [
        r'"([^"]+)"',       # Standard double quotes
        r'\'([^\']+)\'',    # Standard single quotes
    ]
    
    quotes = []
    for pattern in quote_patterns:
        quotes.extend(re.findall(pattern, response))
    
    recommendations["quotes"] = [q.strip() for q in quotes if q.strip()]
    
    # Extract leads (people names with titles/positions)
    lead_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s*[-:,]\s*|\s+is\s+|\s+at\s+)([^,\.\n]+)',
        r'Contact\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[,-]\s*(?:[A-Z][a-z]+\s+)+at\s+'
    ]
    
    leads = []
    for pattern in lead_patterns:
        lead_matches = re.findall(pattern, response)
        for match in lead_matches:
            if isinstance(match, tuple):
                leads.append(f"{match[0]} - {match[1]}")
            else:
                leads.append(match)
    
    recommendations["leads"] = [lead.strip() for lead in leads if lead.strip()]
    
    # Extract events (date patterns with event descriptions)
    event_patterns = [
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})(?:\s*[-:]\s*|\s+)([^,\.\n]+)',
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})(?:\s*[-:]\s*|\s+)([^,\.\n]+)',
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})(?:\s*[-:]\s*|\s+)([^,\.\n]+)'
    ]
    
    events = []
    for pattern in event_patterns:
        event_matches = re.findall(pattern, response)
        for date, description in event_matches:
            events.append(f"{date} - {description}")
    
    # Also look for conference/trade show mentions
    conference_pattern = r'((?:conference|trade show|summit|expo|exhibition|event)(?:\s+in\s+|\s+on\s+|\s+at\s+)(?:[\w\s,]+))(?=\.|,|\n)'
    conference_matches = re.findall(conference_pattern, response, re.IGNORECASE)
    events.extend(conference_matches)
    
    recommendations["events"] = [event.strip() for event in events if event.strip()]
    
    # Store recommendations in session
    if not await request.scope.get("session"):
        await request.scope.update(session={})
    
    session = await request.scope.get("session", {})
    session["latest_recommendations"] = recommendations
    await request.scope.update(session=session)
    
    # Ensure recommendations are returned to the client
    return jsonify({"response": response, "recommendations": recommendations})

@app.route('/test-mic')
async def test_mic():
    """Test endpoint to check microphone setup"""
    logger.info("Microphone test page requested")
    return await render_template('test_mic.html')

async def direct_gemini_call(prompt, temperature=0.8, max_tokens=256):
    """
    Make a direct API call to Gemini 2.0 Flash model.
    Returns the generated text or None if there's an error.
    """
    try:
        # Get API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("No Gemini API key found in environment")
            return None
            
        # Construct API URL and payload
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        logger.info(f"Calling Gemini API with prompt: {prompt[:100]}...")
        
        # Make the API call
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
        
        logger.info(f"Gemini API response status: {response.status_code}")
        
        # Process the response
        if response.status_code == 200:
            data = response.json()
            
            # Extract the generated text
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                    text = candidate["content"]["parts"][0]["text"]
                    logger.info(f"Generated text: {text[:100]}...")
                    return text
            
            logger.error(f"Unexpected response structure: {data}")
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
        
        return None
        
    except Exception as e:
        logger.error(f"Exception in direct_gemini_call: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/test-recommendations')
async def test_recommendations():
    """Simple test endpoint to generate recommendations without flow controller dependencies"""
    try:
        logger.info("Testing recommendations generation")
        
        # Create a simple test recommendation set
        test_recommendations = [
            {
                "name": "Acme Corporation",
                "description": "A global leader in technology solutions",
                "match_score": 95,
                "reason": "Strong match based on your technology interests",
                "website": "https://example.com/acme",
                "key_personnel": [
                    {"name": "John Smith", "title": "CEO"},
                    {"name": "Jane Doe", "title": "CTO"}
                ],
                "recent_news": [
                    {"title": "Acme Announces New Product", "url": "https://example.com/news/1", "date": "2023-06-01", "source": "Tech News"},
                    {"title": "Acme Reports Record Growth", "url": "https://example.com/news/2", "date": "2023-05-15", "source": "Business Weekly"}
                ],
                "events": [
                    {"name": "Acme Annual Conference", "date": "2023-09-15", "location": "San Francisco, CA", "url": "https://example.com/conference"},
                    {"name": "Technology Expo", "date": "2023-10-20", "location": "Chicago, IL", "url": "https://example.com/expo"}
                ],
                "quotes": [
                    {"text": "Acme has revolutionized the way we work", "source": "Industry Expert"},
                    {"text": "Their solutions are best-in-class", "source": "Customer Testimonial"}
                ]
            },
            {
                "name": "TechCorp Inc.",
                "description": "Innovative software solutions for businesses",
                "match_score": 90,
                "reason": "Aligns with your software needs",
                "website": "https://example.com/techcorp",
                "key_personnel": [
                    {"name": "Bob Johnson", "title": "President"},
                    {"name": "Sarah Williams", "title": "VP of Sales"}
                ],
                "recent_news": [
                    {"title": "TechCorp Launches New Platform", "url": "https://example.com/news/3", "date": "2023-06-10", "source": "Tech Today"}
                ],
                "events": [
                    {"name": "Developer Conference", "date": "2023-08-05", "location": "Boston, MA", "url": "https://example.com/devcon"}
                ],
                "quotes": [
                    {"text": "TechCorp's platform is intuitive and powerful", "source": "Industry Review"}
                ]
            }
        ]
        
        # Return the test recommendations
        return jsonify(test_recommendations)
    except Exception as e:
        logger.error(f"Error in test recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "message": "An error occurred while generating test recommendations"
        }), 500

@app.route('/api/has_recommendations', methods=['GET'])
async def has_recommendations():
    """Check if recommendations exist for the current session"""
    try:
        # This is a simple implementation - you might want to enhance this
        # to check actual recommendation availability based on your app's logic
        
        # For now, just return true if flow_controller exists and has keywords
        global flow_controller
        has_recs = (flow_controller is not None and 
                    hasattr(flow_controller, 'keywords') and 
                    len(flow_controller.keywords) > 0)
        
        return jsonify({
            "has_recommendations": has_recs
        })
    except Exception as e:
        logger.error(f"Error checking for recommendations: {str(e)}")
        return jsonify({
            "has_recommendations": False,
            "error": str(e)
        })

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Atom.ai - B2B Research Assistant')
    parser.add_argument('--port', type=int, default=5019, help='Port to run the application on')
    args = parser.parse_args()
    
    port = args.port
    logger.info(f"Starting the application server on 0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port)
