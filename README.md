# Article Analysis Web App

This Flask web application allows users to analyze articles by providing the article URL. It retrieves information such as the title, authors, publish date, summary, sentiment analysis, and associated videos and images.

## Features

- Analyze articles by providing the article URL
- Extract information including title, authors, publish date, and summary
- Perform sentiment analysis on the article's content
- Display associated videos and images

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/scorpionTaj/News-Summarization-System.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

## Usage

1. Access the web application by navigating to `http://localhost:5000/` in your web browser.
2. Enter the URL of the article you want to analyze in the provided input field.
3. Click the "Analyze" button to perform the analysis.
4. View the analysis results including the article information, sentiment analysis, and associated videos and images.

## Technologies Used

- Python
- Flask
- NLTK (Natural Language Toolkit)
- TextBlob
- Newspaper3k

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
