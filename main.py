from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    url_for,
    redirect,
    session,
)
import nltk
from textblob import TextBlob
from newspaper import Article
from newspaper.article import ArticleException
import logging
import os
import functools
from datetime import datetime
import time
import re
import json
from io import BytesIO
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from urllib.parse import urlparse
import validators
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from werkzeug.middleware.proxy_fix import ProxyFix
import hashlib
import bleach
from deep_translator import GoogleTranslator
from PIL import Image
from langdetect import detect, LangDetectException
import pycountry

# Load environment variables from .env file if present
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
    handlers=[
        logging.FileHandler(os.environ.get("LOG_FILE", "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("history", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# NLTK Resources Management
NLTK_RESOURCES = [
    "tokenizers/punkt",
    "taggers/averaged_perceptron_tagger",
    "maxent_ne_chunker/averaged_perceptron_tagger",
    "tokenizers/punkt",
    "corpora/words",
]


def setup_nltk():
    """Ensure all required NLTK resources are available"""
    missing_resources = []

    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing_resources.append(resource)

    if missing_resources:
        logger.info(f"Downloading NLTK resources: {', '.join(missing_resources)}")
        for resource in missing_resources:
            try:
                # Extract component and package name
                package = resource.split("/")[0]
                nltk.download(package)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource}: {e}")
    else:
        logger.info("All NLTK resources already available")


# Setup NLTK resources
setup_nltk()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model")
except OSError:
    logger.warning("Downloading spaCy model")
    try:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        raise

# Configuration
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))  # 10MB
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 3600))  # 1 hour
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 30))  # 30 seconds
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", 60))

# Create and configure Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Add ProxyFix middleware to handle reverse proxy headers correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Rate limiting implementation
request_history = {}


def is_rate_limited(ip_address):
    """Check if the IP address has exceeded rate limits"""
    current_time = time.time()
    if ip_address not in request_history:
        request_history[ip_address] = []

    # Remove requests older than 60 seconds
    request_history[ip_address] = [
        t for t in request_history[ip_address] if current_time - t < 60
    ]

    # Check if under the limit
    if len(request_history[ip_address]) < MAX_REQUESTS_PER_MINUTE:
        request_history[ip_address].append(current_time)
        return False

    return True


@app.before_request
def check_rate_limit():
    """Apply rate limiting before processing requests"""
    # Skip for local development
    if request.remote_addr == "127.0.0.1" or request.remote_addr.startswith("192.168."):
        return

    if is_rate_limited(request.remote_addr):
        logger.warning(f"Rate limit exceeded for IP: {request.remote_addr}")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429


@app.template_filter("intersect")
def intersect_filter(list1, list2):
    if not list1 or not list2:
        return []
    return [item for item in list1 if item in list2]


# Enhanced caching mechanism with TTL support
def disk_cache(dirname, ttl=CACHE_TIMEOUT):
    """
    Decorator for disk-based caching of function results

    Args:
        dirname: Directory to store cache files
        ttl: Time to live in seconds

    Returns:
        Cached function result or new calculation
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key based on function name and arguments
            key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
            cache_file = os.path.join(dirname, f"{key}.json")

            try:
                if os.path.exists(cache_file):
                    # Check if cache entry is still valid
                    file_age = time.time() - os.path.getmtime(cache_file)
                    if file_age < ttl:
                        with open(cache_file, "r") as f:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return json.load(f)
                    else:
                        logger.debug(f"Cache expired for {func.__name__}")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

            # Call the function and cache the result
            result = func(*args, **kwargs)

            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f)
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

            return result

        return wrapper

    return decorator


def validate_url(url):
    """
    Validate if the URL is properly formed and safe

    Args:
        url: URL string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    # Basic URL validation
    if not validators.url(url):
        return False

    # Parse URL to check components
    parsed = urlparse(url)

    # Check for allowed protocols
    if parsed.scheme not in ["http", "https"]:
        return False

    # Additional security checks can be added here
    return True


def sanitize_input(text):
    """
    Sanitize user input to prevent XSS attacks

    Args:
        text: User input text

    Returns:
        str: Sanitized text or empty string if None
    """
    if text is None:
        return ""
    return bleach.clean(text)


def generate_wordcloud(text):
    try:
        # Clean text by removing punctuation and converting to lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())

        # Check if there's enough content
        if len(text.split()) < 3:
            logger.warning("Text too short for wordcloud generation")
            return None

        # Generate the wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
            contour_width=3,
        ).generate(text)

        # Save the wordcloud as an image
        img_data = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_data, format="png", bbox_inches="tight")
        plt.close()

        # Return base64 encoded image
        img_data.seek(0)
        return base64.b64encode(img_data.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}", exc_info=True)
        return None


def extract_named_entities(text):
    try:
        # Process text with spaCy
        doc = nlp(text[:100000])  # Limit text length for performance
        entities = {}

        # Extract and categorize named entities
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)

        # Format entities for display
        formatted_entities = {}
        entity_mapping = {
            "PERSON": "People",
            "ORG": "Organizations",
            "GPE": "Countries/Cities",
            "LOC": "Locations",
            "DATE": "Dates",
            "MONEY": "Financial",
            "PRODUCT": "Products",
        }

        for label, items in entities.items():
            if label in entity_mapping:
                formatted_entities[entity_mapping.get(label, label)] = items[:10]

        return formatted_entities
    except Exception as e:
        logger.error(f"Error extracting named entities: {e}", exc_info=True)
        return {}


def calculate_readability(text):
    try:
        # Calculate readability metrics
        metrics = {
            "flesch_reading_ease": flesch_reading_ease(text),
            "flesch_kincaid_grade": flesch_kincaid_grade(text),
        }

        # Map score to readability level
        readability_levels = [
            (90, "Very Easy"),
            (80, "Easy"),
            (70, "Fairly Easy"),
            (60, "Standard"),
            (50, "Fairly Difficult"),
            (30, "Difficult"),
            (0, "Very Difficult"),
        ]

        for score, level in readability_levels:
            if metrics["flesch_reading_ease"] >= score:
                metrics["readability"] = level
                break
        else:
            metrics["readability"] = "Very Difficult"

        return metrics
    except Exception as e:
        logger.error(f"Error calculating readability: {e}", exc_info=True)
        return {"readability": "Unknown"}


def detect_language(text):
    """
    Detect the language of the provided text

    Args:
        text: Text to analyze

    Returns:
        tuple: (language_code, language_name)
    """
    try:
        if not text or len(text.strip()) < 10:
            return "en", "English"

        # Get sample text for detection (first 1000 chars)
        sample = text[:1000]
        lang_code = detect(sample)

        # Try to get full language name
        try:
            language = pycountry.languages.get(alpha_2=lang_code)
            language_name = language.name if language else lang_code
        except (AttributeError, KeyError):
            # Fallback for languages not in pycountry
            language_name = {
                "ar": "Arabic",
                "zh-cn": "Chinese",
                "nl": "Dutch",
                "en": "English",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "ja": "Japanese",
                "ko": "Korean",
                "pt": "Portuguese",
                "ru": "Russian",
                "es": "Spanish",
            }.get(lang_code, lang_code)

        logger.info(f"Detected language: {language_name} ({lang_code})")
        return lang_code, language_name
    except LangDetectException as e:
        logger.warning(f"Language detection error: {e}")
        return "en", "English"  # Default to English on error
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}", exc_info=True)
        return "en", "English"  # Default to English on error


def translate_text(text, target_language="en", source_language=None):
    """
    Translate text to the target language

    Args:
        text: Text to translate
        target_language: Target language code
        source_language: Source language code (if known)

    Returns:
        str: Translated text
    """
    try:
        # If target is same as source or English text to English target, return original
        if target_language == source_language:
            return text

        # Limit text length to prevent API issues
        limited_text = text[:3000] if len(text) > 3000 else text

        # Auto-detect source language if not specified
        if not source_language:
            source_language, _ = detect_language(text)

        # If target is same as detected source, return original
        if target_language == source_language:
            return text

        # Use specific source language instead of "auto" for better results
        translator = GoogleTranslator(source=source_language, target=target_language)
        result = translator.translate(limited_text)
        return result
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return text


def extract_media_content(html_content):
    """
    Extract video and media content from article HTML

    Args:
        html_content: HTML content of the article

    Returns:
        dict: Extracted media content
    """
    try:
        videos = []
        video_patterns = [
            r'<iframe[^>]*src=[\'"](https?://(?:www\.)?youtube\.com/embed/[^\'"]+)[\'"][^>]*>',
            r'<iframe[^>]*src=[\'"](https?://(?:www\.)?vimeo\.com/video/[^\'"]+)[\'"][^>]*>',
            r'<video[^>]*src=[\'"](https?://[^\'"]+)[\'"][^>]*>',
            r'<source[^>]*src=[\'"](https?://[^\'"]+\.(?:mp4|webm|ogg))[\'"][^>]*>',
        ]

        # Extract videos using regular expressions
        for pattern in video_patterns:
            matches = re.findall(pattern, html_content)
            videos.extend(matches)

        # Check for Twitter embeds
        twitter_embeds = re.findall(
            r'<blockquote class="twitter-tweet"[^>]*>', html_content
        )
        has_twitter = len(twitter_embeds) > 0

        return {"videos": videos[:5], "has_twitter": has_twitter}
    except Exception as e:
        logger.error(f"Error extracting media: {e}", exc_info=True)
        return {"videos": [], "has_twitter": False}


@disk_cache("cache")
def analyze_article(url):
    """
    Analyze an article from a URL

    Args:
        url: URL of the article to analyze

    Returns:
        tuple: (result dict, error message)
    """
    try:
        start_time = time.time()
        logger.info(f"Analyzing article: {url}")

        # Validate URL
        if not validate_url(url):
            return None, "Invalid URL. Please check the format and try again."

        # Download and parse article
        article = Article(url)
        article.download()

        # Extract HTML content and media
        html_content = article.html
        media_content = extract_media_content(html_content)

        # Parse article content
        article.parse()
        article.nlp()

        # Detect language
        article_lang_code, article_lang_name = detect_language(article.text)

        # Sentiment analysis
        analysis = TextBlob(article.text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        # Determine overall sentiment
        if polarity > 0.15:
            sentiment = "Positive"
        elif polarity < -0.15:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Extract keywords
        keywords = article.keywords if hasattr(article, "keywords") else []

        # Generate visualizations and extract entities
        wordcloud_img = generate_wordcloud(article.text)
        entities = extract_named_entities(article.text)
        readability = calculate_readability(article.text)

        # Collect images
        images = []
        if article.top_image:
            images.append(article.top_image)

        if hasattr(article, "images") and article.images:
            for img in article.images:
                if img not in images and len(images) < 5:
                    images.append(img)

        # Count paragraphs
        paragraphs = len(re.split(r"\n\n+", article.text))

        # Assemble result dictionary
        result = {
            "title": article.title,
            "authors": article.authors,
            "publish_date": (
                article.publish_date.strftime("%A %d-%B-%Y")
                if article.publish_date
                else None
            ),
            "summary": article.summary,
            "sentiment": sentiment,
            "polarity": round(polarity * 100, 1),
            "subjectivity": round(subjectivity * 100, 1),
            "images": images,
            "image": article.top_image,
            "text": article.text,
            "keywords": keywords[:10],
            "reading_time": f"{round(len(article.text.split()) / 200, 1)} min",
            "word_count": len(article.text.split()),
            "paragraph_count": paragraphs,
            "wordcloud": wordcloud_img,
            "entities": entities,
            "readability": readability,
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": url,
            "videos": media_content["videos"],
            "has_twitter": media_content["has_twitter"],
            "has_videos": len(media_content["videos"]) > 0,
            "language_code": article_lang_code,
            "language_name": article_lang_name,
        }

        # Save to history
        save_to_history(result)

        # Log processing time
        process_time = time.time() - start_time
        logger.info(f"Processed article in {process_time:.2f} seconds")

        return result, None
    except ArticleException as e:
        logger.error(f"Error processing the article: {e}")
        return (
            None,
            f"Could not download or parse the article. Please check the URL and try again.",
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return (
            None,
            "Failed to connect to the website. Please check your internet connection.",
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return None, f"An unexpected error occurred. Please try again later."


def save_to_history(result):
    """Save analysis result to history"""
    try:
        history_file = os.path.join("history", f"{int(time.time())}.json")
        with open(history_file, "w") as f:
            json_result = result.copy()
            if "text" in json_result:
                json_result["text"] = json_result["text"][:500] + "..."
            if "wordcloud" in json_result:
                del json_result["wordcloud"]
            json.dump(json_result, f)
    except Exception as e:
        logger.error(f"Error saving to history: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    """Home page and article analysis endpoint"""
    result = None
    error = None
    url = None

    if request.method == "POST":
        url = sanitize_input(request.form.get("url", ""))
        result, error = analyze_article(url)

    return render_template("index.html", result=result, error=error, url=url)


@app.route("/analyze", methods=["POST"])
def analyze_ajax():
    """API endpoint for asynchronous article analysis"""
    url = sanitize_input(request.json.get("url", ""))
    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Validate URL format
    if not validate_url(url):
        return jsonify({"error": "Invalid URL format"}), 400

    result, error = analyze_article(url)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({"result": result})


@app.route("/translate", methods=["POST"])
def translate_content():
    """API endpoint for text translation"""
    # First check if the request contains JSON data
    if not request.is_json:
        logger.warning("Translation request without JSON data")
        return jsonify({"error": "Invalid request format - JSON required"}), 400

    data = request.json
    if not data:
        logger.warning("Empty JSON data in translation request")
        return jsonify({"error": "Invalid JSON data"}), 400

    # Get parameters from request with safe defaults
    text = data.get("text", "")
    target = data.get("target", "en")
    source = data.get("source")

    # Sanitize inputs after ensuring they exist
    target = sanitize_input(target)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Detect source language if not provided
    if not source:
        source, source_name = detect_language(text)
    else:
        source = sanitize_input(source)

    try:
        translated = translate_text(text, target, source)
        return jsonify({"translated": translated, "source_language": source})
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return jsonify({"error": "Translation failed", "message": str(e)}), 500


@app.route("/history")
def view_history():
    """View history of analyzed articles"""
    history_files = os.listdir("history")
    history_items = []

    for file in sorted(history_files, reverse=True)[:20]:
        try:
            with open(os.path.join("history", file), "r") as f:
                history_items.append(json.load(f))
        except Exception as e:
            logger.warning(f"Error loading history file {file}: {e}")
            continue

    return render_template("history.html", history=history_items)


@app.route("/compare", methods=["GET", "POST"])
def compare_articles():
    """Compare two articles side by side"""
    if request.method == "POST":
        url1 = sanitize_input(request.form.get("url1", ""))
        url2 = sanitize_input(request.form.get("url2", ""))

        result1, error1 = analyze_article(url1) if url1 else (None, None)
        result2, error2 = analyze_article(url2) if url2 else (None, None)

        return render_template(
            "compare.html",
            result1=result1,
            result2=result2,
            error1=error1,
            error2=error2,
            url1=url1,
            url2=url2,
        )

    return render_template("compare.html")


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return (
        render_template(
            "error.html",
            error_title="Page Not Found",
            error_message="The page you requested could not be found.",
        ),
        404,
    )


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {e}")
    return (
        render_template(
            "error.html",
            error_title="Server Error",
            error_message="An unexpected error occurred. Please try again later.",
        ),
        500,
    )


@app.errorhandler(429)
def too_many_requests(e):
    """Handle rate limiting errors"""
    return (
        render_template(
            "error.html",
            error_title="Too Many Requests",
            error_message="You've made too many requests. Please try again later.",
        ),
        429,
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("DEBUG", "False").lower() == "true")
