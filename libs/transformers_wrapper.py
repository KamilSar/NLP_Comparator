# Używamy Twojego pipeline NER (angielski conll03). Bez stopwords (brak natywnych).
import re
from transformers import pipeline, AutoTokenizer

_ner_pipeline = None
_tokenizer = None

def preload_models():
    global _ner_pipeline, _tokenizer
    try:
        model_name = "xlm-roberta-large-finetuned-conll03-english"
        _ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Błąd ładowania modelu Transformers: {e}")
        _ner_pipeline = None
        _tokenizer = None

def process_text(text, lang_code=None):
    # Jeśli model niezaładowany – zwróć puste struktury (uczciwy obraz)
    if _ner_pipeline is None or _tokenizer is None:
        print("Model Transformers nie został załadowany.")
        return {
            "Tokens": [],
            "Lemmas": [],
            "Entities": [],
            "Sentences": [],
            "POS": [],
            # brak stopwords
        }

    try:
        # NER
        ner_result = _ner_pipeline(text)
        Entities = [ent["word"] for ent in ner_result if ent.get("entity_group", "O") != "O"]
        # Tokeny – prosto, żeby było spójnie z UI (to NIE jest tokenizacja modelu)
        Tokens = text.split()
    except Exception as e:
        print(f" Błąd Transformers: {e}")
        Tokens = []
        Entities = []

    # Prosta SBD
    Sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
    POS = []      # brak POS
    Lemmas = []   # brak lematyzacji

    return {
        "Tokens": Tokens,
        "Lemmas": Lemmas,
        "Entities": Entities,
        "Sentences": Sentences,
        "POS": POS,
        # brak stopwords (biblioteka ich nie ma)
    }
