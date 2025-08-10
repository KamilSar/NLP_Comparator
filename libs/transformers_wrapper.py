from transformers import pipeline, AutoTokenizer

_ner_pipeline = None
_tokenizer = None

def preload_models():
    global _ner_pipeline, _tokenizer
    try:
        model_name = "xlm-roberta-large-finetuned-conll03-english"
        _ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple"
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Błąd ładowania modelu Transformers: {e}")
        _ner_pipeline = None
        _tokenizer = None

def process_text(text, lang=None):
    if _ner_pipeline is None or _tokenizer is None:
        print("Model Transformers nie został załadowany.")
        return {
            "tokens": [],
            "lemmas": [],
            "entities": [],
        }

    try:
        result = _ner_pipeline(text)
        # ✔️ Wydobywamy byty nazwane
        entities = [ent["word"] for ent in result if ent.get("entity_group", "O") != "O"]
        # ✔️ Używamy prostszej tokenizacji — spójnej z innymi bibliotekami
        tokens = text.split()
    except Exception as e:
        print(f"Błąd podczas przetwarzania tekstu przez Transformers: {e}")
        tokens = []
        entities = []

    return {
        "tokens": tokens,
        "lemmas": [],  # brak lematyzacji
        "entities": entities,
    }
