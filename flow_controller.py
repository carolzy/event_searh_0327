from typing import Dict, List, Optional
import logging
import os
import random
import httpx
import json
from datetime import datetime
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowController:
    """Controls the multi-step B2B sales flow"""
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the flow controller."""
        # Load API keys
        load_dotenv()
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"Loaded Gemini API key: {self.gemini_api_key[:10] if self.gemini_api_key else 'Not found'}")
        
        # User data
        self.current_product_line = ""
        self.current_sector = ""
        self.current_segment = ""
        self.keywords = []
        self.linkedin_consent = False
        self.zip_code = ""
        
        # Conversation memory
        self.conversation_memory = []
        self.context_summary = ""
        
        # Flow state
        self.steps = [
            'product',
            'market',
            'company_size',
            'linkedin_consent',
            'zip_code',
            'complete'
        ]
    
    def get_next_step(self, current_step):
        """Get the next step in the flow."""
        try:
            current_index = self.steps.index(current_step)
            next_index = current_index + 1
            if next_index < len(self.steps):
                return self.steps[next_index]
            else:
                return 'complete'
        except ValueError:
            # If the current step is not found, start from the beginning
            return self.steps[0]
    
    async def get_question(self, step: str) -> str:
        """Get the question for the current step."""
        if step == 'complete':
            return "Great! We've completed your setup. Let's find some target companies for you!"
        
        # Define the default questions for each step with better examples
        default_questions = {
            'product': "What product or service does your company offer? Please provide a brief description.",
            'market': "Which industries or sectors are you targeting with your product or service?",  # Improved market question
            'company_size': "What size of companies are you primarily targeting? (e.g., Small, Medium, Enterprise)",
            'linkedin_consent': "Would you like to connect your LinkedIn account to enhance your company recommendations?",
            'zip_code': "What zip code are you in? This will help us find relevant local events. (You can skip this question if you prefer.)",
        }
        
        # Define example answers for each step
        example_answers = {
            'product': "We offer a data quality platform for machine learning teams.",
            'market': "We target tech companies and enterprise businesses in healthcare and finance.",
            'company_size': "We focus on mid-market and enterprise companies with over 500 employees.",
            'linkedin_consent': "Yes, I'd like to connect my LinkedIn account.",
            'zip_code': "94105",
        }
        
        try:
            # Use the Gemini API to generate a conversational question
            if not self.gemini_api_key:
                logger.warning("No Gemini API key found. Using default questions.")
                question = default_questions.get(step, "Tell me more about your needs.")
            else:
                # Generate a more conversational question based on the current context
                prompt = f"""
                You are a friendly B2B research assistant helping a user set up their company research preferences.
                
                Current context:
                - Product/Service: {self.current_product_line or 'Not provided yet'}
                - Target Market: {self.current_sector or 'Not provided yet'}
                - Company Size: {self.current_segment or 'Not provided yet'}
                
                We need to ask about: {step}
                
                Default question: {default_questions.get(step, "Tell me more about your needs.")}
                
                Generate a friendly, conversational question for this step. Keep it concise (1-2 sentences max).
                Do not use technical jargon. Make it sound natural and engaging.
                Do not include any thinking process in your response.
                """
                
                question = await self._call_gemini_api(prompt)
                
            # Add example answer hint
            example = example_answers.get(step, "")
            if example:
                question += f"\n\nSpeak or type your answer. For example: '{example}'"
            
            return question
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return self._get_fallback_question(f"We need to ask about: {step}")
    
    async def get_follow_up_question(self, step: str, previous_answer: str, follow_up_count: int = 0, suggest_next: bool = False) -> str:
        """Get a follow-up question for the current step."""
        try:
            # Check if we should suggest moving to the next step
            if suggest_next:
                next_step = self.get_next_step(step)
                next_step_name = {
                    'product': 'your target market',
                    'market': 'your target company size',
                    'company_size': 'LinkedIn integration',
                    'linkedin_consent': 'your location',
                    'zip_code': 'completing your setup'
                }.get(next_step, 'the next step')
                
                return f"Thanks for that information. Would you like to add anything else or shall we move on to {next_step_name}?"
            
            # Check for signs of user impatience in the previous answer
            impatience_indicators = [
                "next", "move on", "continue", "skip", "enough", "let's go", 
                "proceed", "done", "finished", "complete", "that's it", "that's all"
            ]
            
            if any(indicator in previous_answer.lower() for indicator in impatience_indicators):
                next_step = self.get_next_step(step)
                return f"Let's move on to the next question. {await self.get_question(next_step)}"
            
            # Use the Gemini API to generate a follow-up question
            if not self.gemini_api_key:
                logger.warning("No Gemini API key found. Using default follow-up questions.")
                return "Can you tell me more about that?"
            
            # Generate a more conversational follow-up question based on the current context
            prompt = f"""
            You are a friendly B2B research assistant helping a user set up their company research preferences.
            
            Current context:
            - Product/Service: {self.current_product_line or 'Not provided yet'}
            - Target Market: {self.current_sector or 'Not provided yet'}
            - Company Size: {self.current_segment or 'Not provided yet'}
            
            Current step: {step}
            User's answer: "{previous_answer}"
            Follow-up count: {follow_up_count + 1}
            
            Generate a brief, friendly follow-up question that helps clarify or expand on their answer.
            Keep it concise (1 sentence max). Do not use technical jargon.
            If this is the second follow-up (count = 1), make it a final clarification before moving on.
            Do not include any thinking process in your response.
            """
            
            follow_up = await self._call_gemini_api(prompt)
            
            return follow_up
                
        except Exception as e:
            logger.error(f"Error generating follow-up question: {str(e)}")
            return "Can you tell me more about that?"
    
    async def _call_gemini_api(self, prompt):
        """Call the Gemini API with the given prompt."""
        if not self.gemini_api_key:
            logger.error("Gemini API key not found")
            return self._get_fallback_question(prompt)
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=data,
                    timeout=5.0  # Reduced timeout to improve latency
                )
                
                if response.status_code != 200:
                    logger.error(f"Error from Gemini API: {response.status_code}, {response.text}")
                    return self._get_fallback_question(prompt)
                
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        return content["parts"][0]["text"]
                
                logger.error(f"Unexpected response format from Gemini API: {result}")
                return self._get_fallback_question(prompt)
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return self._get_fallback_question(prompt)
    
    async def _call_llm(self, prompt):
        """Alias for _call_gemini_api for backward compatibility"""
        logger.info("_call_llm called, forwarding to _call_gemini_api")
        return await self._call_gemini_api(prompt)
    
    def _get_fallback_question(self, prompt):
        """Get a fallback question if the Gemini API call fails."""
        step = "unknown"
        lines = prompt.split('\n')
        if any("We need to ask about:" in line for line in lines):
            step_line = [line for line in lines if "We need to ask about:" in line]
            if step_line:
                step = step_line[0].split(':')[1].strip()
        
        fallback_questions = {
            'product': "What product or service are you selling?",
            'market': "Which market or industry are you targeting?",
            'company_size': "What size of companies are you focusing on?",
            'keywords': "Here are some keywords I've generated based on your input. Would you like to edit them?",
            'linkedin_consent': "Would you like to include LinkedIn profiles in your research?",
            'zip_code': "What's your zip code for location-based insights? (Type 'skip' to skip this step)",
            'unknown': "What else would you like to tell me about your needs?"
        }
        
        return fallback_questions.get(step, fallback_questions['unknown'])
        
    async def _generate_keywords_with_llm(self, context: str) -> List[str]:
        """Generate relevant keywords for the current context using Gemini API"""
        try:
            # Get the Gemini API key
            if not self.gemini_api_key:
                logger.error("No Gemini API key found. Using fallback keyword generation.")
                return ["B2B", "Sales", "Marketing", "Technology", "Innovation"]
            
            # Prepare the prompt for keyword generation
            prompt = f"""
            <think>
            Generate keywords that describe the product offering and target customer based on the following context:
            {context}
            
            Keep updating them to be a more accurate set as you receive more information.
            For example, if the product is "Databricks", keywords might include "cloud transformation", "data analytics", "big data", etc.
            </think>
            
            Generate only the most relevant keywords that would help find ideal target companies.
            Format your response as a comma-separated list of keywords only, without any additional text or explanations.
            """
            
            # Log the API request for debugging
            logger.info(f"Calling Gemini API for keyword generation with context: {context}")
            
            # Make the API request
            keywords = await self._call_gemini_api(prompt)
            
            # Parse the keywords from the response
            keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            
            # Limit to 20 keywords as per user request
            keywords_list = keywords_list[:20]
            
            logger.info(f"Generated keywords with LLM: {keywords_list}")
            return keywords_list
                
        except Exception as e:
            logger.error(f"Error generating keywords with LLM: {str(e)}")
            return ["B2B", "Sales", "Marketing", "Technology", "Innovation"]

class QuestionEngine:
    """Generates dynamic questions for the onboarding flow"""
    
    def __init__(self):
        """Initialize the question engine with templates"""
        self.default_questions = {
            "greeting": "Atom AI really would like to know more about the product you are selling.",
            "product": "What problem does your product solve?",
            "market": "Which market sector or industry are you targeting?",
            "differentiation": "How do you plan to differentiate your product offering compared to the market?",
            "company_size": "What size companies are your ideal customers?",
            "location": "What is your location (zip code or area like New York City or San Francisco Bay Area) so we can recommend local events to meet your buyers?",
            "linkedin": "If you'd like, please make sure you are logged into your LinkedIn via Atom so that we can find more precise recommendations based on your connections!",
            "additional": "Any specific keywords that describe your ideal customers?"
        }
        
        self.templates = {
            "greeting": [
                "Atom AI really would like to know more about the product you are selling.",
                "Atom AI really would like to know more about the product you are selling.",
                "Atom AI really would like to know more about the product you are selling."
            ],
            "product": [
                "What problem does your product solve?",
                "How do you define your product?",
                "Tell me what problem does your product solve?"
            ],
            "market": [
                "Which industries or markets need this solution most?",
                "Based on your {product}, which sectors should we prioritize?",
                "For companies struggling with {product}, which verticals matter most?"
            ],
            "company_size": [
                "What size companies are you targeting? (Enterprise: 5000+, Mid-Market: 100-5000, SMB: <100)",
                "For {market} companies, which size range fits best?",
                "Should we focus on enterprise giants, mid-market movers, or SMB innovators?"
            ],
            "zip_code": [
                "What's your zip code? This helps us find nearby events (optional - say 'skip' to skip this step).",
                "To find networking events near you, what's your zip code? (Say 'skip' if you prefer not to share).",
                "For {market} events in your area, what's your zip code? (Optional - say 'skip' to continue)."
            ],
            "additional": [
                "Any specific keywords that describe your ideal customers?",
                "What pain points or buying signals should I watch for?",
                "Any other criteria that makes a prospect perfect for you?"
            ]
        }
    
    def generate(self, step, context):
        """Generate a question based on the current step and context"""
        if step == "greeting":
            greeting = self.default_questions["greeting"]
            product_q = self.default_questions["product"]
            return f"{greeting} {product_q}"
            
        if step not in self.templates:
            return self.default_questions.get(step, "Tell me more about your business.")
            
        templates = self.templates[step]
        
        # Try to find a template that can use the context
        for template in templates:
            if "{" in template:
                try:
                    return template.format(**context)
                except KeyError:
                    continue
                    
        # Fall back to a random template without context
        return random.choice(templates)
