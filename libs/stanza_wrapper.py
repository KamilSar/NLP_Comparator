import stanza

_loaded = {}

def _ensure_pipeline(lang_code="pl"):
    if lang_code not in _loaded:
        stanza.download(lang_code, verbose=False)
        _loaded[lang_code] = stanza.Pipeline(
            lang=lang_code,
            processors="tokenize,mwt,pos,lemma,depparse,ner",
            use_gpu=False
        )
    return _loaded[lang_code]

def process_text(text, lang_code="pl"):
    """
    Stanza: brak własnych stopwords- NIE zwracamy żadnych pól stopwords.
    Zwracamy SBD, POS, Lemmas, Entities.
    """
    nlp = _ensure_pipeline(lang_code)
    doc = nlp(text)

    Tokens, Lemmas, Entities = [], [], []
    Sentences, POS = [], []

    for sent in doc.sentences:
        sent_text = " ".join(w.text for w in sent.words)
        Sentences.append(sent_text)
        for w in sent.words:
            Tokens.append(w.text)
            Lemmas.append(w.lemma)
            POS.append((w.text, w.upos))  # Universal POS

    for ent in doc.ents:
        Entities.append((ent.text, ent.type))

    return {
        "Tokens": Tokens,
        "Lemmas": Lemmas,
        "Entities": Entities,
        "Sentences": Sentences,
        "POS": POS,
        # brak stopwords pól, bo biblioteka ich nie ma
    }
