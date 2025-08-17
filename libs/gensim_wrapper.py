import re
from gensim.utils import simple_preprocess
from langdetect import detect

def process_text(text, lang_code=None):
    """
    Gensim: brak natywnych stopwords/POS/NER.
    - Tokenizacja: simple_preprocess (lowercase, bez interpunkcji)
    - SBD: prosta regexowa (kropka, wykrzyknik, pytajnik)
    - POS: N/A (puste)
    - Stopwords: brak (NIE UŻYWAMY fallbacku!)
    """
    Tokens = simple_preprocess(text)
    lang = lang_code or detect(text)

    Sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
    POS = []         # brak
    Entities = []    # brak
    Lemmas = Tokens  # brak lematyzacji- zwracamy tokeny

    return {
        "Tokens": Tokens,
        "Lemmas": Lemmas,
        "Entities": Entities,
        "Sentences": Sentences,
        "POS": POS,
        # brak stopwords pól- brak w bibliotece
        "morph": [],
        "dependencies": [],
        "segment_words": Tokens,
    }
