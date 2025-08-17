import spacy
from spacy.cli import download

SUPPORTED_MODELS = {
    "pl": "pl_core_news_sm",
    "en": "en_core_web_sm",
}

_loaded = {}

def _get_nlp(lang_code):
    if lang_code not in SUPPORTED_MODELS:
        raise ValueError(f"Nieobsługiwany język: {lang_code}")
    model_name = SUPPORTED_MODELS[lang_code]
    if lang_code not in _loaded:
        try:
            _loaded[lang_code] = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' nie znaleziony — pobieram...")
            download(model_name)
            _loaded[lang_code] = spacy.load(model_name)
    return _loaded[lang_code]

def _native_stopwords(nlp):
    return set(getattr(nlp.Defaults, "stop_words", set()))

def process_text(text, lang_code):
    """
    spaCy: ma własne stopwords- zwracamy TokensNoStop i StopwordsRemoved.
    """
    nlp = _get_nlp(lang_code)
    doc = nlp(text)

    Tokens = [t.text for t in doc]
    Lemmas = [t.lemma_ for t in doc]
    Entities = [(ent.text, ent.label_) for ent in doc.ents]
    Sentences = [s.text.strip() for s in doc.sents]
    POS = [(t.text, t.pos_) for t in doc]

    sw = _native_stopwords(nlp)
    TokensNoStop = [t for t in Tokens if t.lower() not in sw]
    StopwordsRemoved = len(Tokens) - len(TokensNoStop)

    morph = [(t.text, t.morph.to_dict()) for t in doc]
    dependencies = [(t.text, t.dep_, t.head.text) for t in doc]
    segment_words = [t.text for t in doc]

    return {
        "Tokens": Tokens,
        "Lemmas": Lemmas,
        "Entities": Entities,
        "Sentences": Sentences,
        "POS": POS,
        "TokensNoStop": TokensNoStop,
        "StopwordsRemoved": StopwordsRemoved,
        "morph": morph,
        "dependencies": dependencies,
        "segment_words": segment_words,
    }
