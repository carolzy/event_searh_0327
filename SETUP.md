# Setup Instructions for Atom.ai Voice Search

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Steps

1. Clone this repository:
   ```
   git clone https://github.com/YOUR_USERNAME/atom-voice-search.git
   cd atom-voice-search
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file from the template:
   ```
   cp env.template .env
   ```

5. Edit the `.env` file and add your API keys:
   - GEMINI_API_KEY
   - PERPLEXITY_API_KEY
   - OPENAI_API_KEY
   - ELEVENLABS_API_KEY

## Running the Application

Start the application with:
```
python app.py
```

The application will be available at http://localhost:8080 by default.

You can specify a different port with:
```
python app.py --port 8081
```

## Features

- Voice-based interaction for company research
- Dynamic keyword generation
- Company recommendations based on user preferences
- Detailed company insights and analysis

## Troubleshooting

If you encounter any issues:
1. Make sure all API keys are correctly set in the `.env` file
2. Check that all required directories exist (data/user_journey, cache)
3. Ensure you're using a compatible Python version
