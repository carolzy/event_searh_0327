
import os
import logging
import httpx
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FlowController:
    """Controls the onboarding flow and manages user input/state."""

    def __init__(self):
        # User-provided data
        self.current_product_line = ""
        self.current_website = ""
        self.product_differentiation = ""
        self.current_sector = ""
        self.current_segment = ""
        self.zip_code = ""
        self.linkedin_consent = False
        self.keywords: List[str] = []

        # Internal state
        self.conversation_memory: List[str] = []

        # Define step order
        self.steps = [
            "product",
            "website",
            "differentiation",
            "market",
            "company_size",
            "location",
            "linkedin",
            "complete"
        ]
        self.current_step_index = 0

        # API key for Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def get_next_step(self, current_step: str) -> str:
        """Return the next step in the flow after the current step."""
        try:
            index = self.steps.index(current_step)
            if index < len(self.steps) - 1:
                return self.steps[index + 1]
            return "complete"
        except ValueError:
            return self.steps[0]  # default to first step if invalid

    async def get_question(self, step: str) -> str:
        """Return a dynamic or fallback question for the given step."""
        fallback_questions = {
            "product": "What product are you working on or selling?",
            "website": "Do you have a website where I can learn more?",
            "differentiation": "How is your product different from existing solutions?",
            "market": "Which market or industry are you focused on?",
            "company_size": "What size of companies are you targeting?",
            "location": "What’s your zip code or city for local opportunities?",
            "linkedin": "Would you like to connect your LinkedIn to personalize recommendations?",
            "complete": "Awesome! Let me now generate your ideal company recommendations."
        }

        if step == "complete":
            return fallback_questions["complete"]

        if not self.gemini_api_key:
            logger.warning("Gemini API key not found, using fallback question.")
            return fallback_questions.get(step, "Tell me more.")

        try:
            prompt = self._build_question_prompt(step)
            question = await self._call_gemini_api(prompt)
            return question or fallback_questions.get(step, "Tell me more.")
        except Exception as e:
            logger.error(f"Gemini error for step {step}: {e}")
            return fallback_questions.get(step, "Tell me more.")

    def _build_question_prompt(self, step: str) -> str:
        """Create a Gemini prompt based on current step and context."""
        context_lines = [
            f"Product: {self.current_product_line or 'N/A'}",
            f"Market: {self.current_sector or 'N/A'}",
            f"Company Size: {self.current_segment or 'N/A'}"
        ]

        return f"""
        You are Atom, a friendly B2B onboarding assistant.

        Context:
        {chr(10).join(context_lines)}

        Now generate a friendly, concise (1-2 sentences) and natural sounding question for this step: {step}

        The question should be clear, engaging, and appropriate for a business founder.
        Don't include your thought process in the output — only return the question.
        """

    async def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Send prompt to Gemini Flash and return the generated text."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 256
            }
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
        return None

    def get_hint_for_step(self, step: str) -> str:
        """Provide a short example or tip for each step."""
        hints = {
            "product": "e.g., 'We help companies monitor AI model performance.'",
            "website": "e.g., 'https://mystartup.com'",
            "differentiation": "e.g., 'Unlike legacy tools, ours uses real-time signals.'",
            "market": "e.g., 'Healthtech, fintech, or logistics startups'",
            "company_size": "e.g., 'SMBs with 10-500 employees'",
            "location": "e.g., '94103' or 'San Francisco Bay Area'",
            "linkedin": "e.g., 'Yes' or 'Not right now'",
            "complete": "You're all set!"
        }
        return hints.get(step, "")

    def get_product(self): return self.current_product_line
    def get_market(self): return self.current_sector
    def get_company_size(self): return self.current_segment
    def get_location(self): return self.zip_code
    def get_keywords(self): return self.keywords
    def get_linkedin_consent(self): return self.linkedin_consent

    async def store_answer(self, step: str, answer: str):
        """Store answer in the appropriate field."""
        if step == "product":
            self.current_product_line = answer
        elif step == "website":
            self.current_website = answer
        elif step == "differentiation":
            self.product_differentiation = answer
        elif step == "market":
            self.current_sector = answer
        elif step == "company_size":
            self.current_segment = answer
        elif step == "location":
            self.zip_code = answer
        elif step == "linkedin":
            self.linkedin_consent = answer.lower() in ["yes", "true", "sure"]
        self.conversation_memory.append(f"{step}: {answer}")

    async def process_voice_answer(self, transcription: str) -> str:
        """Store current voice answer and move to next step."""
        current_step = self.steps[self.current_step_index]
        await self.store_answer(current_step, transcription)
        self.current_step_index += 1
        return self.steps[self.current_step_index] if self.current_step_index < len(self.steps) else "complete"

    async def clean_keywords(self) -> List[str]:
        """Generate cleaned keywords for recommendation."""
        if not self.gemini_api_key:
            return []
        context = f"{self.current_product_line} {self.product_differentiation} {self.current_sector}"
        return await self.generate_keywords(context)
    async def generate_keywords(self, context: str) -> List[str]:
        """Generate keywords using Gemini based on user context."""
        if not self.gemini_api_key:
            logger.warning("No Gemini API key for keyword generation.")
            return []

        prompt = f"""
        Based on this context, generate relevant B2B targeting keywords:

        {context}

        Format your response as a comma-separated list of keywords only.
        """

        try:
            response = await self._call_gemini_api(prompt)
            return [kw.strip() for kw in response.split(',') if kw.strip()]
        except Exception as e:
            logger.error(f"Keyword generation failed: {e}")
            return []

    def reset(self):
        """Reset all user-provided fields and memory."""
        self.current_product_line = ""
        self.current_website = ""
        self.product_differentiation = ""
        self.current_sector = ""
        self.current_segment = ""
        self.zip_code = ""
        self.linkedin_consent = False
        self.keywords = []
        self.conversation_memory = []
        self.current_step_index = 0


