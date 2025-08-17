# TextBlob nie ma własnych stopwords/POS/NER dla PL/EN.
# Zostawiamy tokenizację/lematy z TextBlob, SBD proste, POS = N/A.
import re
from textblob import TextBlob

def process_text(text, lang_code=None):
    try:
        blob = TextBlob(text)
        Tokens = list(blob.words)
        # Proste lematy dostępne w TextBlob (angielski działa sensownie; dla PL są słabe)
        Lemmas = [w.lemmatize() for w in Tokens]
        Entities = []  # TextBlob nie wspiera NER
        # Prosta segmentacja zdań (SBD)
        Sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
        POS = []  # brak POS
        # Dodatkowe pola, jeśli chcesz spójność z innymi
        morph = []
        dependencies = []
        segment_words = Tokens
    except Exception as e:
        print(f" Błąd TextBlob: {e}")
        Tokens = []
        Lemmas = []
        Entities = []
        Sentences = []
        POS = []
        morph = []
        dependencies = []
        segment_words = []

    return {
        "Tokens": Tokens,
        "Lemmas": Lemmas,
        "Entities": Entities,
        "Sentences": Sentences,
        "POS": POS,
        "morph": morph,
        "dependencies": dependencies,
        "segment_words": segment_words,
    }
