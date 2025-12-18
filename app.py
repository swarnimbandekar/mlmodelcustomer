from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (only first time)
nltk.download('vader_lexicon')

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
    score = sia.polarity_scores(text)['compound']
    if score < -0.05:
        return "Negative"
    elif score > 0.05:
        return "Positive"
    else:
        return "Neutral"


def assign_priority(sentiment, text):
    urgent_keywords = [
        "delay", "delayed", "lost", "damaged",
        "urgent", "no response", "not delivered"
    ]

    if sentiment == "Negative" and any(k in text.lower() for k in urgent_keywords):
        return "High"
    elif sentiment == "Negative":
        return "Medium"
    else:
        return "Low"


def smart_category(text):
    """
    Rule-based detection first (high confidence),
    fallback to ML model if no rule matches
    """
    t = text.lower()

    if any(k in t for k in ["delivery", "delayed", "post", "parcel", "courier", "order"]):
        return "Delivery"

    if any(k in t for k in ["bill", "amount", "charged", "payment", "refund", "money"]):
        return "Billing"

    if any(k in t for k in ["support", "service", "rude", "staff", "helpdesk"]):
        return "Service"

    if any(k in t for k in ["internet", "network", "connection", "slow"]):
        return "Technical"

    # ML fallback
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


def generate_response(category, priority):
    if category == "Delivery":
        return "We apologize for the delay. Your delivery complaint has been registered and marked as high priority."

    if category == "Billing":
        return "We are reviewing your billing concern and will update you shortly."

    if category == "Service":
        return "We regret the inconvenience. Our support team will contact you shortly."

    if category == "Technical":
        return "Our technical team is looking into the issue and will resolve it as soon as possible."

    return "Thank you for contacting us. We are reviewing your complaint."


# ===============================
# API endpoint
# ===============================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "complaint_text" not in data:
        return jsonify({"error": "complaint_text is required"}), 400

    text = data["complaint_text"]

    category = smart_category(text)
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
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
