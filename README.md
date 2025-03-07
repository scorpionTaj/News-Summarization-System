# News Summarization System

A web application that analyzes news articles to provide summaries, sentiment analysis, entity recognition, and visualizations.

## Features

- **Article Summarization**: Automatically generates concise summaries of news articles
- **Sentiment Analysis**: Determines if an article has a positive, negative, or neutral tone
- **Named Entity Recognition**: Identifies people, organizations, locations, and other entities
- **Readability Metrics**: Calculates how easy or difficult the article is to read
- **Keyword Extraction**: Identifies the main topics and keywords from the article
- **Word Cloud Generation**: Visual representation of frequent words in the article
- **Multi-Language Support**: Automatically detects article language and provides translation options
- **Article Comparison**: Compare two articles side by side to analyze differences
- **Media Content Extraction**: Extracts videos, Twitter embeds and other media from articles
- **History Tracking**: Keep a record of previously analyzed articles
- **Dark/Light Mode**: Adjustable visual theme for comfortable reading

## Recent Updates

- **Language Auto-Detection**: System now automatically detects the language of articles
- **Enhanced Translation**: Improved translation between multiple languages with better source language handling
- **Robust Error Handling**: Better error handling for API requests and content processing
- **Input Sanitization**: Improved security with comprehensive input validation
- **Rate Limiting**: Protection against excessive requests
- **Caching System**: Disk-based caching with TTL for better performance

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

4. Configure environment variables by creating a `.env` file:

```
DEBUG=False
SECRET_KEY=your_secret_key
LOG_LEVEL=INFO
CACHE_TIMEOUT=3600
MAX_REQUESTS_PER_MINUTE=60
```

## Usage

Run the application:

```bash
python main.py
```

Then open your browser and navigate to `http://localhost:5000`

## API Endpoints

- **`/analyze`**: POST endpoint for asynchronous article analysis
- **`/translate`**: POST endpoint for content translation
- **`/history`**: GET endpoint to view analysis history
- **`/compare`**: GET/POST endpoint to compare articles

## Configuration Options

The application can be configured using environment variables or a `.env` file:

- `DEBUG`: Enable debug mode (True/False)
- `SECRET_KEY`: Secret key for Flask session
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `CACHE_TIMEOUT`: Cache timeout in seconds
- `MAX_CONTENT_LENGTH`: Maximum content length in bytes
- `MAX_REQUESTS_PER_MINUTE`: Rate limiting threshold
- `REQUEST_TIMEOUT`: Timeout for external requests in seconds

## Dependencies

- Flask - Web framework
- NLTK - Natural Language Processing
- TextBlob - Sentiment analysis
- newspaper3k - Article extraction
- spaCy - Named entity recognition
- deep-translator - Translation services
- wordcloud - Word cloud generation
- langdetect - Language detection
- pycountry - Language code mapping

## License

MIT
