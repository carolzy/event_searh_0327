
import os
import logging
import re
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class QuestionEngine:
    """Generates onboarding questions using Gemini 2.0 Flash or fallback templates."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.steps = ['product', 'market', 'differentiation', 'company_size', 'linkedin', 'location', 'complete']

        self.templates = {
            'product': "What product or service do you sell?",
            'market': "What market or industry do you target?",
            'differentiation': "What makes your product unique compared to competitors?",
            'company_size': "What size companies do you typically sell to? (e.g., SMB, Mid-Market, Enterprise)",
            'linkedin': "Would you like to connect your LinkedIn account to improve recommendations?",
            'location': "What zip code are you in? This helps us suggest local events. (You can skip this)",
            'complete': "Awesome! I've gathered everything I need. Let’s find some great companies for you."
        }

    def get_next_step(self, current_step):
        """Returns the next step in the onboarding flow."""
        try:
            index = self.steps.index(current_step)
            return self.steps[index + 1] if index + 1 < len(self.steps) else 'complete'
        except ValueError:
            return 'product'

    async def get_question(self, current_step, context=None):
        """Returns a question for the given step."""
        if not current_step:
            return self.templates.get('complete')

        if self.api_key:
            try:
                return await self._generate_with_gemini(current_step, context)
            except Exception as e:
                logger.error(f"Gemini error for step {current_step}: {str(e)}")

        return self.templates.get(current_step, "Can you tell me more about that?")

    def _build_prompt(self, step, context):
        """Builds an LLM prompt based on step and context."""
        product = context.get('product', '')
        market = context.get('market', '')
        differentiation = context.get('differentiation', '')
        company_size = context.get('company_size', '')
        zip_code = context.get('location', '')

        prompt = f"You are Atom.ai, a helpful B2B sales assistant.\n"
        prompt += f"You're onboarding a user and collecting sales info. The current step is '{step}'.\n\n"
        prompt += f"Known info so far:\n"
        if product: prompt += f"- Product: {product}\n"
        if market: prompt += f"- Market: {market}\n"
        if differentiation: prompt += f"- Differentiation: {differentiation}\n"
        if company_size: prompt += f"- Company Size: {company_size}\n"
        if zip_code: prompt += f"- Zip Code: {zip_code}\n\n"

        prompt += f"Generate a short, friendly question for the step: '{step}'. Be conversational and concise.\n"
        return prompt

    async def _generate_with_gemini(self, step, context):
        """Calls Gemini Flash 2.0 API to generate the question."""
        prompt = self._build_prompt(step, context or {})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.9,
                "maxOutputTokens": 256
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            content = result.get("candidates", [{}])[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                question = self._clean_response(parts[0].get("text", ""))
                return question

        logger.warning(f"Fallback to template: {step}")
        return self.templates.get(step, "Tell me more about your goals.")

    def _clean_response(self, text):
        """Cleans LLM response to make sure it's a proper question."""
        text = re.sub(r'^(Q|Question):\s*', '', text.strip(), flags=re.IGNORECASE)
        if not text.endswith('?'):
            text += '?'
        return text

    async def generate_greeting(self) -> str:
        prompt = """
        Generate a short (1-2 sentence) friendly greeting introducing Atom.ai,
        a helpful assistant for identifying B2B sales opportunities.
        The tone should be natural, like a knowledgeable friend.
        End by asking what product the user sells.
        """
        try:
            response = await self._generate_custom_prompt(prompt)
            return response or "Hi! I'm Atom.ai — here to help with your B2B sales research. What product are you selling?"
        except Exception as e:
            logger.error(f"Greeting generation failed: {e}")
            return "Hi! I'm Atom.ai. I really look forward to learning about your product and also about you! So first tell me what problems does your product solve?"

    async def _generate_custom_prompt(self, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 128
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            content = result.get("candidates", [{}])[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return self._clean_response(parts[0].get("text", ""))
        logger.warning("Gemini greeting fallback used.")
        return None