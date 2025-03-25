# Atom.ai Voice Search Application

This folder contains the organized essential files for the Atom.ai voice search application, a daily company research assistant that provides deep insights for B2B professionals.

## Core Files

- `app.py` - Main application file with the Quart web server and API endpoints
- `company_recommender.py` - Handles company recommendations based on user preferences
- `flow_controller.py` - Manages the conversation flow and user journey
- `question_engine.py` - Processes questions and generates responses using Gemini API
- `recommendation_verifier.py` - Verifies the quality of company recommendations
- `user_memory.py` - Manages user preferences and memory
- `voice_processor.py` - Handles text-to-speech conversion

## Templates

The `templates` directory contains HTML templates for the web interface:
- `index.html` - Main chat interface
- `onboarding.html` - Onboarding flow
- `recommendations.html` - Company recommendations display
- And other supporting templates

## Static Assets

The `static` directory contains:
- CSS files for styling
- JavaScript files for client-side functionality
- Favicon and other assets

## Configuration

- `requirements.txt` - Python dependencies

## Target Users

- B2B Sales Professionals
- VCs/Investors
- Journalists
- Job Seekers

## Core User Flow

1. Daily Company Focus (Duolingo-style)
   - Company essence and differentiators
   - Current year's investment focus and resource allocation

2. Optional Deep Dive into Key Personnel
   - Recent news article quotes (2-year window)
   - Interview summaries
   - Contextual snippets based on user keywords

3. Dynamic Keyword Generation
   - User provides their context (e.g., "I sell Delta Lake at Databricks")
   - System infers relevant keywords (e.g., "Cloud Transformation", "Cloud Storage")
