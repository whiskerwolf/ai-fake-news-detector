import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
        and len(word) > 2
    ]

    return " ".join(words)