
import os
import time
import logging
import base64
from dotenv import load_dotenv
from quart import Quart, render_template, request, jsonify, send_file

from flow_controller import FlowController
from voice_processor import VoiceProcessor
from question_engine import QuestionEngine
from company_recommender import CompanyRecommender

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Quart app
app = Quart(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Version for cache busting
VERSION = str(int(time.time()))

# Stopwords
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

# Initialize core components
flow_controller = FlowController()
voice_processor = VoiceProcessor(flow_controller)
question_engine = QuestionEngine()
company_recommender = CompanyRecommender(flow_controller)

@app.route("/")
async def index():
    flow_controller.reset()
    greeting = await question_engine.get_question("product")
    first_question = await flow_controller.get_question("product")
    return await render_template("index.html", greeting=greeting, first_question=first_question, version=VERSION)

@app.route("/api/onboarding", methods=["POST"])
async def onboarding_step():
    data = await request.get_json()
    step = data.get("step")
    answer = data.get("answer", "")
    logger.info(f"Onboarding: {step} => {answer}")
    
    await flow_controller.store_answer(step, answer)
    next_step = flow_controller.get_next_step(step)
    
    if next_step == "complete":
        cleaned_keywords = await flow_controller.clean_keywords()
        recommendations = await company_recommender.generate_recommendations()
        return jsonify({
            "success": True,
            "completed": True,
            "keywords": cleaned_keywords,
            "recommendations": recommendations
        })

    question = await flow_controller.get_question(next_step)
    audio_data = await voice_processor.text_to_speech(question)
    return jsonify({
        "success": True,
        "step": next_step,
        "question": question,
        "audio": audio_data
    })

@app.route("/api/get_question", methods=["GET"])
async def get_question():
    step = request.args.get("step", "product")
    question = await flow_controller.get_question(step)
    audio_data = await voice_processor.text_to_speech(question)
    return jsonify({
        "success": True,
        "question": question,
        "audio": audio_data,
        "keywords": flow_controller.keywords
    })

@app.route("/api/recommendations", methods=["GET"])
async def get_recommendations():
    recs = await company_recommender.generate_recommendations()
    return jsonify(recs)

@app.route("/api/text_to_speech", methods=["POST"])
async def tts():
    data = await request.get_json()
    text = data.get("text", "")
    audio_data = await voice_processor.text_to_speech(text)
    return jsonify({"audio": audio_data})

@app.route("/api/voice_interaction", methods=["POST"])
async def voice_interaction():
    if not voice_processor:
        return jsonify({"error": "Voice processor not initialized"}), 500

    try:
        data = await request.get_json()
        text = data.get("text")
        step = data.get("step", "product")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        await flow_controller.store_answer(step, text)
        next_step = flow_controller.get_next_step(step)

        if next_step == "complete":
            cleaned_keywords = await flow_controller.clean_keywords()
            recommendations = await company_recommender.generate_recommendations()
            return jsonify({
                "success": True,
                "completed": True,
                "text": "You're all set! Generating your results.",
                "keywords": cleaned_keywords,
                "recommendations": recommendations,
                "show_recommendations_tab": True
            })

        question = await flow_controller.get_question(next_step)
        audio = await voice_processor.text_to_speech(question)

        return jsonify({
            "success": True,
            "text": question,
            "next_step": next_step,
            "audio": audio
        })

    except Exception as e:
        logger.error(f"Voice interaction failed: {str(e)}")
        return jsonify({"error": "Voice processing failed"}), 500

@app.route("/onboarding_data.csv")
async def download_onboarding_data():
    return await send_file("onboarding_data.csv", as_attachment=True)

@app.after_request
async def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
    
@app.route("/recommendations")
async def recommendations_page():
    return await render_template("recommendations.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5019)
