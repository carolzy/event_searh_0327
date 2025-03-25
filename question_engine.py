import random
import logging
import re
import os
import requests
import json
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class QuestionEngine:
    """Generates dynamic questions for the onboarding flow based on context"""
    
    def __init__(self):
        """Initialize the QuestionEngine with default templates"""
        self.logger = logging.getLogger(__name__)
        
        # Check if Gemini API key is available
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            self.logger.info("Gemini API key found. Using dynamic LLM-based questions.")
        else:
            self.logger.warning("No Gemini API key found. Using static question templates.")
        
        # Default question templates
        self.question_templates = {
            'product': "What product or service does your company offer? Please provide a brief description.",
            'market': "What market or industry sector are you targeting with your product or service?",
            'company_size': "What size of companies are you primarily targeting? (e.g., Small, Medium, Enterprise)",
            'keywords': "Cool! We've generated some target keywords for your product/service. You can edit, remove, or add to these keywords below:",
            'linkedin_consent': "Would you like to connect your LinkedIn account to enhance your company recommendations? This will help us find more relevant matches based on your professional network.",
            'zip_code': "What zip code are you in? This will help us find relevant local events. (You can skip this question if you prefer.)"
        }
    
    def generate(self, step, context=None):
        """
        Generate a question based on the current step and context
        
        Args:
            step (str): The current step in the onboarding flow
            context (dict): Context from previous answers
            
        Returns:
            str: A dynamically generated question
        """
        if context is None:
            context = {}
            
        if step is None:
            return "Great! We've completed your setup. Let's find some target companies for you!"
        
        # Always try to use the LLM first if we have an API key
        if self.gemini_api_key:
            try:
                llm_response = self._generate_with_llm(step, context)
                if llm_response:
                    return llm_response
            except Exception as e:
                logger.error(f"Error using LLM for question generation: {str(e)}")
                # Fall through to template-based questions if LLM fails
        
        # Fallback to template-based questions
        if step == 'product':
            return self._generate_product_question(context)
        elif step == 'market':
            return self._generate_market_question(context)
        elif step == 'company_size':
            return self._generate_company_size_question(context)
        elif step == 'zip_code':
            return self._generate_zip_code_question(context)
        elif step == 'linkedin_consent':
            return self._generate_linkedin_consent_question(context)
        elif step == 'additional':
            return self._generate_additional_question(context)
        else:
            return "Tell me more about your needs."
    
    def _generate_with_llm(self, step, context):
        """Generate a question using the Gemini API"""
        try:
            # Construct a prompt based on the current step and context
            prompt = self._construct_prompt(step, context)
            
            # Call the Gemini API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(
                url,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        question = content["parts"][0]["text"]
                        
                        # Clean up the response to ensure it's a single question
                        question = self._clean_llm_response(question)
                        
                        logger.info(f"Generated question with Gemini API: {question}")
                        return question
                
                logger.error(f"Unexpected response format from Gemini API: {result}")
                return self._fallback_question(step, context)
            else:
                logger.error(f"Error calling Gemini API: {response.status_code} - {response.text}")
                # Fall back to template-based questions
                return self._fallback_question(step, context)
                
        except Exception as e:
            logger.error(f"Error generating question with LLM: {str(e)}")
            # Fall back to template-based questions
            return self._fallback_question(step, context)
    
    def _construct_prompt(self, step, context):
        """Construct a prompt for the LLM based on the current step and context"""
        product = context.get('product', '')
        market = context.get('market', '')
        company_size = context.get('company_size', '')
        
        if step == 'product':
            return """
            Generate a friendly, conversational question asking what product or service the user sells.
            Keep it short and engaging.
            """
        elif step == 'market':
            return f"""
            The user sells: {product}
            
            Generate a friendly, conversational question asking what industry or market sector they target.
            Reference their product/service in your question.
            Keep it short and engaging.
            """
        elif step == 'company_size':
            return f"""
            The user sells: {product}
            They target the {market} industry.
            
            Generate a friendly, conversational question asking what size of companies they typically target 
            (e.g., small businesses, mid-market, enterprise).
            Reference their product/service and industry in your question.
            Keep it short and engaging.
            """
        elif step == 'zip_code':
            return f"""
            The user sells: {product}
            They target the {market} industry.
            They focus on {company_size} companies.
            
            Generate a friendly, conversational question asking for their zip code to provide location-based recommendations.
            Mention that this is optional and they can skip this step if they prefer.
            Keep it short and engaging.
            """
        elif step == 'additional':
            return f"""
            The user sells: {product}
            They target the {market} industry.
            They focus on {company_size} companies.
            
            Generate a friendly, conversational question asking for any additional context about their product/service
            that would help find better company matches. Suggest they could mention specific use cases,
            features, or benefits that would make a company a good fit.
            Keep it short and engaging.
            """
        else:
            return "Generate a friendly, conversational question asking for more information."
    
    def _clean_llm_response(self, response):
        """Clean up the LLM response to ensure it's a single question"""
        # Remove any extra quotes
        response = response.strip('"\'')
        
        # Remove any prefixes like "Question: "
        response = re.sub(r'^(Question|Q):\s*', '', response, flags=re.IGNORECASE)
        
        # Ensure it ends with a question mark if it's a question
        if not response.endswith('?') and not response.endswith('.'):
            response += '?'
            
        return response
    
    def _fallback_question(self, step, context):
        """Fallback to template-based questions if LLM fails"""
        if step == 'product':
            return self._generate_product_question(context)
        elif step == 'market':
            return self._generate_market_question(context)
        elif step == 'company_size':
            return self._generate_company_size_question(context)
        elif step == 'zip_code':
            return self._generate_zip_code_question(context)
        elif step == 'additional':
            return self._generate_additional_question(context)
        else:
            return "Tell me more about your needs."
    
    def _generate_product_question(self, context):
        """Generate a question about the product"""
        return self.question_templates['product']
    
    def _generate_market_question(self, context):
        """Generate a question about the target market based on product info"""
        return self.question_templates['market']
    
    def _generate_company_size_question(self, context):
        """Generate a question about company size based on previous answers"""
        return self.question_templates['company_size']
    
    def _generate_zip_code_question(self, context):
        """Generate a question about the user's zip code"""
        return self.question_templates['zip_code']
    
    def _generate_additional_question(self, context):
        """Generate a question about additional keywords based on previous answers"""
        return self.question_templates['keywords']
    
    def _generate_linkedin_consent_question(self, context):
        """Generate a question asking for LinkedIn login consent"""
        return self.question_templates['linkedin_consent']
    
    def _extract_product_type(self, product_description):
        """Extract the product type from the description"""
        product_types = ['platform', 'solution', 'software', 'service', 'tool', 'application', 'system', 'product']
        
        for product_type in product_types:
            if product_type in product_description:
                return product_type
        
        return "solution"  # Default fallback
    
    async def get_next_question(self, current_step, context):
        """
        Get the next question based on the current step and context
        
        Args:
            current_step (str): The current step in the onboarding flow
            context (dict): Context from previous answers
            
        Returns:
            str: A dynamically generated question for the next step
        """
        # Determine the next step based on the current step
        next_step = self._get_next_step(current_step)
        
        # Build comprehensive context for the LLM
        full_context = {
            'product': context.get('product', ''),
            'market': context.get('market', ''),
            'company_size': context.get('company_size', ''),
            'keywords': context.get('keywords', []),
            'linkedin_consent': context.get('linkedin_consent', ''),
            'zip_code': context.get('zip_code', ''),
            'current_step': current_step,
            'next_step': next_step
        }
        
        logger.info(f"Getting next question for step: {next_step} with context: {full_context}")
        
        # Always try to use the LLM first if we have an API key
        if self.gemini_api_key:
            try:
                # Construct a conversational prompt based on the user's previous answers
                prompt = f"""
                The user is in an onboarding flow for a B2B research assistant called 'Deal Detective'.
                
                Current information we have:
                - Product/Service: {full_context['product']}
                - Market/Industry: {full_context['market']}
                - Target Company Size: {full_context['company_size']}
                - Keywords: {', '.join(full_context['keywords']) if isinstance(full_context['keywords'], list) else full_context['keywords']}
                - LinkedIn Consent: {full_context['linkedin_consent']}
                - Zip Code: {full_context['zip_code']}
                
                The user just answered a question about: {current_step}
                
                Now, we need to ask about: {next_step}
                
                Generate a friendly, conversational question for the {next_step} step.
                Make it engaging and personalized based on their previous answers.
                Keep it concise (1-2 sentences maximum).
                """
                
                # Call the Gemini API
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
                        timeout=10.0
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        content = result["candidates"][0]["content"]
                        if "parts" in content and len(content["parts"]) > 0:
                            question = content["parts"][0]["text"]
                            
                            # Clean up the response to ensure it's a single question
                            question = self._clean_llm_response(question)
                            
                            logger.info(f"Generated question with Gemini API: {question}")
                            return question
                    
                    logger.error(f"Unexpected response format from Gemini API: {result}")
                    return self._fallback_question(next_step, full_context)
                else:
                    logger.error(f"Error calling Gemini API: {response.status_code} - {response.text}")
                    # Fall back to template-based questions
                    return self._fallback_question(next_step, full_context)
                    
            except Exception as e:
                logger.error(f"Error generating question with LLM: {str(e)}")
                # Fall back to template-based questions
                return self._fallback_question(next_step, full_context)
        
        # Fallback to template-based questions if no API key or if LLM fails
        return self._fallback_question(next_step, full_context)
    
    def _get_next_step(self, current_step):
        """Determine the next step based on the current step"""
        step_sequence = ['product', 'market', 'company_size', 'keywords', 'linkedin_consent', 'zip_code']
        
        try:
            current_index = step_sequence.index(current_step)
            if current_index < len(step_sequence) - 1:
                return step_sequence[current_index + 1]
            else:
                return 'complete'
        except ValueError:
            # If current_step is not in the sequence, start from the beginning
            return step_sequence[0]
    
    def _fallback_question(self, step, context):
        """Generate a fallback question based on templates"""
        if step == 'complete':
            return "Great! We've completed your setup. Let's find some target companies for you!"
            
        if step in self.question_templates:
            return self.question_templates[step]
        
        return "Tell me more about your needs."

    async def get_gemini_response(self, prompt):
        """
        Get a response from the Gemini Flash 2.0 API
        
        Args:
            prompt (str): The prompt to send to the Gemini API
            
        Returns:
            str: The response text from the Gemini API
        """
        try:
            if not self.gemini_api_key:
                logger.error("No Gemini API key found. Cannot generate response.")
                return None
                
            # Call the Gemini API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ],
                "tools": [
                    {
                        "googleSearchRetrieval": {}
                    }
                ]
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
                        response_text = content["parts"][0]["text"]
                        logger.info(f"Generated response with Gemini API: {response_text[:100]}...")
                        return response_text
                
                logger.error(f"Unexpected response format from Gemini API: {result}")
                return None
            else:
                logger.error(f"Error calling Gemini API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini API: {str(e)}")
            return None
