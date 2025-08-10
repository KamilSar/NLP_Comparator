from gensim.utils import simple_preprocess
from langdetect import detect

def process_text(text, lang_code=None):
    """
    Przetwarza tekst za pomocą Gensim (prosta tokenizacja).
    Nie używa żadnych zewnętrznych lematyzatorów (np. NLTK).
    """
    tokens = simple_preprocess(text)
    lang = lang_code or detect(text)

    return {
        "tokens": tokens,
        "lemmas": tokens,         # brak lematyzacji – zwracamy tokeny
        "entities": [],           # brak NER
        "morph": [],              # brak morfologii
        "dependencies": [],       # brak składni
        "segment_words": tokens   # segmentacja = tokeny
    }
