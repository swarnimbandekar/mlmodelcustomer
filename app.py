from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (first run only)
nltk.download("vader_lexicon")

app = Flask(__name__)

# ===============================
# Load trained ML models
# ===============================
model = pickle.load(open("complaint_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sentiment analyzer
sia = SentimentIntensityAnalyzer()

# ===============================
# Helper functions
# ===============================

def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score < -0.05:
        return "Negative"
    elif score > 0.05:
        return "Positive"
    else:
        return "Neutral"


def assign_priority(sentiment, text):
    urgent_keywords = [
        "delay", "delayed", "lost", "damaged",
        "urgent", "no response", "not delivered",
        "missing", "stuck"
    ]

    if sentiment == "Negative" and any(k in text.lower() for k in urgent_keywords):
        return "High"
    elif sentiment == "Negative":
        return "Medium"
    else:
        return "Low"


def predict_category(text):
    """
    ML-first prediction (PRIMARY)
    """
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


def generate_response(category, priority):
    responses = {
        "Delivery Issues": "We apologize for the delay. Your delivery-related complaint has been registered and is being addressed.",
        "Billing / Payment": "We are reviewing your billing or payment concern and will update you shortly.",
        "Service / Staff": "We regret the inconvenience caused by our staff or service. The issue has been forwarded for action.",
        "Technical": "Our technical team is reviewing the issue and will resolve it as soon as possible.",
        "Other": "Thank you for contacting us. Your complaint has been registered for further review."
    }

    response = responses.get(category, responses["Other"])

    if priority == "High":
        response += " This complaint has been marked as high priority."

    return response


# ===============================
# API endpoint
# ===============================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "complaint_text" not in data:
        return jsonify({"error": "complaint_text is required"}), 400

    text = data["complaint_text"]

    category = predict_category(text)
    sentiment = get_sentiment(text)
    priority = assign_priority(sentiment, text)
    response = generate_response(category, priority)

    return jsonify({
        "category": category,
        "sentiment": sentiment,
        "priority": priority,
        "ai_response": response
    })


# ===============================
# Run server
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
