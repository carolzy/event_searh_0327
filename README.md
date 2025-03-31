# Atom.ai

> *Connecting visionaries with their perfect audience*

## The Art of Modern Connection

Every social interaction with a purpose is a selling moment. (It might sound cynical at first, but think of the upside!)

Selling extends beyond goods or services—it's about selling yourself, your brand, your vision. In today's rapidly evolving business landscape, making the right connections is everything.

## Why Atom.ai Matters Now

AI is reshaping business from the ground up. Soon, we'll see multi-billion-dollar companies built by "nuclear teams" of fewer than five people.

Since the ChatGPT breakthrough, the surge of new startups has been staggering—many with a B2B focus, making sales a critical daily task for founders.

And it's not just founders:
- VCs "sell" to secure deals with top startups
- Job candidates pitch themselves to employers
- Companies compete to attract the best talent

**Selling isn't just about transactions—it's about persuasion, influence, and mutual gain.**

I have had much fun designing this application: we are at the mere beginning of seeing AI transforming businesses to be centered around meaningful human interactions and valuating true contribution from the innovative, the builder and the real visionaries!

## The Atom.ai User Flow
<img width="573" alt="Screenshot 2025-03-30 at 9 58 38 PM" src="https://github.com/user-attachments/assets/0f4d97e2-c634-41c3-8f09-42610ca12630" />


1. Discovery Conversation**: Atom.ai engages with you to understand the core of your product/service offering in a voice-enabled conversational onboarding flow. 
2. Insight Generation**: Keywords about your product/service are extracted as the conversation concludes
3. Strategic Event Matching**: Receive recommendations for events where potential buyers will be present
4. Company Intelligence**: Get insights about companies where these potential connections work
5. Compatibility Analysis**: Learn why these companies are ideal matches as buyers
6. Voice of Leadership**: Access quotes from recent interviews with leaders at these companies

## Architecture

### Core Files

- `app.py` - Main application file with the Quart web server and API endpoints
- `company_recommender.py` - Handles company recommendations based on user preferences
- `flow_controller.py` - Manages the conversation flow and user journey
- `question_engine.py` - Processes questions and generates responses using Gemini API
- `recommendation_verifier.py` - Verifies the quality of company recommendations
- `user_memory.py` - Manages user preferences and memory
- `voice_processor.py` - Handles text-to-speech conversion

### Templates

The `templates` directory contains HTML templates for the web interface:
- `index.html` - Main chat interface
- `onboarding.html` - Onboarding flow
- `recommendations.html` - Company recommendations display
- And other supporting templates

### Static Assets

The `static` directory contains:
- CSS files for styling
- JavaScript files for client-side functionality
- Favicon and other assets

### Configuration

- `requirements.txt` - Python dependencies

## Who We Serve

- **Primary Persona**: Startup founders in San Francisco and New York
  - *Specific focus*: AI startup founders
  - *Niche segment*: LLM Observability startups
- **B2B Sales Professionals** looking to identify ideal prospects
- **VCs/Investors** seeking promising connections
- **Journalists** researching industry trends and key players
- **Job Seekers** targeting companies aligned with their expertise

## Engineering Innovation

1. We employ a dynamic approach to match your offerings with the optimal events where you'll find future buyers
2. Our streamlined database structure eliminates the need to store all events/company/leader/article/quote recommendations at the user level
3. The system matches the "memory vector" generated from each conversation with the most compatible events, creating a personalized experience

---

*Atom.ai: Where meaningful connections become transformative opportunities.*
