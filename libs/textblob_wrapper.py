# libs/textblob_wrapper.py
from textblob import TextBlob

def process_text(text):
    try:
        blob = TextBlob(text)
        tokens = blob.words
        lemmas = [word.lemmatize() for word in tokens]
        entities = []  # TextBlob nie wspiera NER
        morph = [(word, {}) for word in tokens]  # Brak morfologii
        dependencies = []  # Brak zależności składniowych
    except Exception as e:
        print(f"❌ Błąd TextBlob: {e}")
        tokens, lemmas, entities, morph, dependencies = [], [], [], [], []

    return {
        "tokens": list(tokens),
        "lemmas": lemmas,
        "entities": entities,
        "morph": morph,
        "dependencies": dependencies,
        "segment_words": list(tokens),
    }