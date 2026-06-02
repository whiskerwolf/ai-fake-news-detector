from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# --------------------------
# Load Dataset
# --------------------------

fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Labels
fake_df["label"] = 0
real_df["label"] = 1

# --------------------------
# Balance Dataset
# --------------------------

min_size = min(
    len(fake_df),
    len(real_df)
)

fake_df = fake_df.sample(
    min_size,
    random_state=42
)

real_df = real_df.sample(
    min_size,
    random_state=42
)

# --------------------------
# Combine & Shuffle
# --------------------------
# Combine fake + real dataset
df = pd.concat([fake_df, real_df], ignore_index=True)

# Load additional real news examples
extra_real = pd.read_csv("data/extra_real_news.csv")
extra_real["label"] = 1

# Give higher importance to modern real news
extra_real = pd.concat([extra_real] * 10, ignore_index=True)

df = pd.concat([df, extra_real], ignore_index=True)

extra_fake = pd.read_csv("data/extra_fake_news.csv")
extra_fake["label"] = 0

extra_fake = pd.concat([extra_fake] * 10, ignore_index=True)

df = pd.concat([df, extra_real, extra_fake], ignore_index=True)

# VERY IMPORTANT
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# --------------------------
# Create Content
# --------------------------

df["content"] = (
    df["title"].fillna("") +
    " " +
    df["text"].fillna("")
)

# --------------------------
# Add Modern Real News
# --------------------------

extra_real = pd.read_csv(
    "data/extra_real_news.csv"
)

extra_real["title"] = ""
extra_real["text"] = extra_real["text"]

extra_real["content"] = (
    extra_real["title"] +
    " " +
    extra_real["text"]
)

df = pd.concat(
    [df, extra_real],
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X = df["content"]
y = df["label"]

print("\nDataset Balance:")
print(y.value_counts())

# --------------------------
# TF-IDF Vectorizer
# --------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

X_vectorized = vectorizer.fit_transform(X)

# --------------------------
# Train/Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------
# Model
# --------------------------

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=0.7,
    solver="liblinear",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# --------------------------
# Evaluation
# --------------------------

preds = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    preds
)

print(f"\nAccuracy: {accuracy:.4f}")

# --------------------------
# Save Model
# --------------------------

joblib.dump(
    model,
    "models/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")