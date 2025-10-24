
# Libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle


# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Text Preprocessing
def preprocess_comment(comment):
    """Clean, tokenize, and lemmatize a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return comment


# Load Model & Vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise


# Initialize model and vectorizer once
model, vectorizer = load_model_and_vectorizer("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")


# Routes 
@app.route('/')
def home():
    return "Welcome to the YouTube Sentiment Analysis Flask API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed).toarray()
        predictions = model.predict(transformed).tolist()

        response = [{"comment": c, "sentiment": p} for c, p in zip(comments, predictions)]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        texts = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess & vectorize
        preprocessed = [preprocess_comment(text) for text in texts]
        transformed = vectorizer.transform(preprocessed).toarray()

        # Predict
        predictions = model.predict(transformed).tolist()

        response = [
            {"comment": text, "sentiment": pred, "timestamp": ts}
            for text, pred, ts in zip(texts, predictions, timestamps)
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# Run App
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
