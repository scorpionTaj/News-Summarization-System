from flask import Flask, render_template, request
import nltk
from textblob import TextBlob
from newspaper import Article
from newspaper.article import ArticleException
import logging

nltk.download("punkt")

app = Flask(__name__)


def analyze_article(url):
    """
    Function to analyze a given article URL.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        analysis = TextBlob(article.text)
        sentiment = (
            "Positive"
            if analysis.polarity > 0
            else "Negative" if analysis.polarity < 0 else "Neutral"
        )

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
            "image": article.top_image,
            "text": article.text,
        }

        return result, None
    except ArticleException as e:
        logging.error(f"Error processing the article: {e}")
        return None, f"Error processing the article: {e}"
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None, f"An unexpected error occurred: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        url = request.form["url"]
        result, error = analyze_article(url)
        return render_template("index.html", result=result, error=error, url=url)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
