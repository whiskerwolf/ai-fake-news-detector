import joblib
from preprocess import clean_text

# ---------------------------
# Load Model & Vectorizer
# ---------------------------

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_news(news_text):
    """
    Predict whether news is Fake or Real
    """

    # ---------------------------
    # Input Validation
    # ---------------------------

    if not news_text or len(news_text.strip()) < 10:
        return {
            "error": "Please enter a meaningful news headline or article."
        }

    cleaned_text = clean_text(news_text)

    if len(cleaned_text.split()) < 2:
        return {
            "error": "Input is too short to analyze properly."
        }

    # ---------------------------
    # Transform Text
    # ---------------------------

    text_vector = vectorizer.transform([cleaned_text])

    # ---------------------------
    # Prediction
    # ---------------------------

    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    fake_prob = round(probabilities[0] * 100, 2)
    real_prob = round(probabilities[1] * 100, 2)

    confidence = max(fake_prob, real_prob)

    # ---------------------------
    # Better Confidence Logic
    # ---------------------------

    if confidence >= 75:
        confidence_level = "High Confidence"
    elif confidence >= 60:
        confidence_level = "Moderate Confidence"
    else:
        confidence_level = "Low Confidence"


    if confidence >= 65:
        result = prediction
    else:
        result = "⚠️ Uncertain Result — Model confidence is low."

    # ---------------------------
    # Smarter Result Logic
    # ---------------------------

    gap = abs(fake_prob - real_prob)

    if gap < 12:
        result = "Uncertain Result — Mixed signals detected."
    elif prediction == 1:
        result = "Likely Real News"
    else:
        result = "Likely Fake News"

    # ---------------------------
    # Prediction Basis
    # ---------------------------

    reasons = []

    if "uncertain" in result.lower():
        reasons.append(
            "The model found mixed credibility signals."
        )
        reasons.append(
            "The headline may need more context."
        )

    elif prediction == 1:
        reasons.append(
            "Text structure resembles real news reporting."
        )
        reasons.append(
            "Moderate credibility indicators detected."
        )

    else:
        reasons.append(
            "The wording pattern resembles misleading content."
        )
        reasons.append(
            "Credibility signals appear weaker."
        )

    reasons.append(
        f"Confidence Level: {confidence_level}"
    )

    # ---------------------------
    # Return Output
    # ---------------------------

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "confidence_level": confidence_level,
        "reasons": reasons
    }