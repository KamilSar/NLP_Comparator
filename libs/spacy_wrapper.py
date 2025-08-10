import spacy
from spacy.cli import download

# Obsługiwane modele językowe
SUPPORTED_MODELS = {
    "pl": "pl_core_news_sm",
    "en": "en_core_web_sm"
}

# Cache na załadowane modele
loaded_models = {}

def load_spacy_model(lang_code):
    """
    Ładuje model spaCy dla danego języka.
    Jeśli model nie jest zainstalowany — pobiera go automatycznie.
    """
    if lang_code not in SUPPORTED_MODELS:
        raise ValueError(f"Nieobsługiwany język: {lang_code}")

    model_name = SUPPORTED_MODELS[lang_code]

    if lang_code not in loaded_models:
        try:
            loaded_models[lang_code] = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' nie znaleziony — trwa pobieranie...")
            download(model_name)
            loaded_models[lang_code] = spacy.load(model_name)

    return loaded_models[lang_code]

def process_text(text, lang_code):
    """
    Przetwarza tekst za pomocą spaCy.
    Zwraca tokeny, lematy, byty NER, morfologię, składnię, segmentację.
    """
    nlp = load_spacy_model(lang_code)
    doc = nlp(text)

    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    morph = [(token.text, token.morph.to_dict()) for token in doc]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    segment_words = [token.text for token in doc]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "entities": entities,
        "morph": morph,
        "dependencies": dependencies,
        "segment_words": segment_words
    }
