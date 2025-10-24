
# Libraries imports
import io
import os
import re
import logging
import pickle

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud

import pandas as pd

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt-sentiment-api")


# ---------- Ensure required NLTK data ----------
def ensure_nltk_data():
    try:
        stopwords.words('english')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        logger.info("Downloading missing NLTK data (stopwords, wordnet, omw-1.4)...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    except Exception as e:
        logger.exception("Error ensuring NLTK data: %s", e)
        raise


ensure_nltk_data()


# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)


# ---------- Preprocessing ----------
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

def preprocess_comment(comment):
    """Clean and lemmatize a comment safely."""
    try:
        if comment is None:
            return ""
        text = str(comment).lower().strip()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^a-z0-9\s!?.,]', '', text)
        tokens = [w for w in text.split() if w not in STOP_WORDS]
        lems = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(lems)
    except Exception as e:
        logger.warning("Preprocessing error: %s", e)
        return ""


# ---------- Model loading ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "./lgbm_model.pkl")
VECT_PATH = os.environ.get("VECT_PATH", "./tfidf_vectorizer.pkl")

def load_model_and_vectorizer(model_path, vectorizer_path):
    logger.info("Loading model from %s and vectorizer from %s", model_path, vectorizer_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info("Model and vectorizer loaded successfully.")
    return model, vectorizer

try:
    model, vectorizer = load_model_and_vectorizer(MODEL_PATH, VECT_PATH)
except Exception as e:
    logger.exception("Failed to load model/vectorizer: %s", e)
    model, vectorizer = None, None


# ---------- Routes ----------
@app.route("/")
def home():
    return "YouTube Sentiment Analysis API (Flask) - Running"


# ----- Predict Sentiment -----
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model/vectorizer not loaded on server."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "comments" not in data:
        return jsonify({"error": "Missing 'comments' field."}), 400

    comments = data["comments"]
    if not isinstance(comments, list) or not comments:
        return jsonify({"error": "'comments' must be a non-empty list."}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        try:
            preds = model.predict(transformed)
        except Exception:
            preds = model.predict(transformed.toarray())
        preds = [int(p) if hasattr(p, "item") else p for p in preds]
        return jsonify([{"comment": c, "sentiment": p} for c, p in zip(comments, preds)])
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ----- Predict with timestamps -----
@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model/vectorizer not loaded on server."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "comments" not in data:
        return jsonify({"error": "Missing 'comments' field."}), 400

    comments_data = data["comments"]
    if not isinstance(comments_data, list):
        return jsonify({"error": "'comments' must be a list of objects."}), 400

    try:
        texts = [item.get("text", "") for item in comments_data]
        timestamps = [item.get("timestamp") for item in comments_data]

        preprocessed = [preprocess_comment(t) for t in texts]
        transformed = vectorizer.transform(preprocessed)
        try:
            preds = model.predict(transformed)
        except Exception:
            preds = model.predict(transformed.toarray())

        preds = [int(p) if hasattr(p, "item") else p for p in preds]
        response = [
            {"comment": t, "sentiment": p, "timestamp": ts}
            for t, p, ts in zip(texts, preds, timestamps)
        ]
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction with timestamps error: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ----- Generate Pie Chart -----
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json(force=True)
        sentiment_counts = data.get("sentiment_counts", {})
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "white"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Error generating chart: %s", e)
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# ----- Generate Wordcloud -----
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json(force=True)
        comments = data.get("comments", [])
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=STOP_WORDS,
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Error generating wordcloud: %s", e)
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# ----- Generate Sentiment Trend Graph -----
@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json(force=True)
        sentiment_data = data.get("sentiment_data", [])
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        if "timestamp" not in df or "sentiment" not in df:
            return jsonify({"error": "Missing required keys (timestamp, sentiment)."}), 400

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        plt.figure(figsize=(12, 6))
        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker="o",
                linestyle="-",
                label=labels[val],
                color=colors[val],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Error generating trend graph: %s", e)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
