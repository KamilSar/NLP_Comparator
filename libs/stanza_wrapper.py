import stanza

loaded_pipelines = {}

def process_text(text, lang_code="pl"):
    if lang_code not in loaded_pipelines:
        stanza.download(lang_code, verbose=False)
        loaded_pipelines[lang_code] = stanza.Pipeline(
            lang=lang_code,
            processors="tokenize,mwt,pos,lemma,depparse,ner",
            use_gpu=False
        )

    nlp = loaded_pipelines[lang_code]
    doc = nlp(text)

    tokens = []
    lemmas = []
    entities = []

    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append(word.text)
            lemmas.append(word.lemma)

    for ent in doc.ents:
        entities.append(ent.text)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "entities": entities,
    }
