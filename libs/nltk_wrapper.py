import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords

# Pobieramy ZASOBY NLTK – to nadal "wewnątrz NLTK", nie korzystamy z innych bibliotek
def _safe_download(pkg):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

for pkg in [
    "punkt", "wordnet", "omw-1.4",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "stopwords",
]:
    _safe_download(pkg)

lemmatizer = WordNetLemmatizer()

def _lang_name(lang_code: str) -> str:
    return "polish" if str(lang_code).lower().startswith("pl") else "english"

def _wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def _native_stopwords(lang_code: str):
    try:
        return set(stopwords.words(_lang_name(lang_code)))
    except Exception:
        # brak stopwords w lokalnym NLTK → nie wyświetlamy sekcji w app (bo pola będą puste)
        return None

def process_text(text, lang_code="pl"):
    lang_name = _lang_name(lang_code)

    # Tokeny
    try:
        Tokens = word_tokenize(text, language=lang_name)
    except Exception:
        Tokens = text.split()

    # Zdania (SBD)
    try:
        Sentences = sent_tokenize(text, language=lang_name)
    except Exception:
        Sentences = []

    # POS i Lematy
    POS = []
    Lemmas = []
    if lang_name == "english":
        try:
            POS = pos_tag(Tokens)
            Lemmas = [WordNetLemmatizer().lemmatize(tok, _wordnet_pos(pos)) for tok, pos in POS]
        except Exception:
            POS = []
            Lemmas = []
    else:
        # dla PL: NLTK nie zapewnia sensownego POS/lemmaty → puste (uczciwy obraz)
        POS = []
        Lemmas = []

    # Stopwords (tylko natywne NLTK; jeśli brak, nie zwracamy tych pól)
    sw = _native_stopwords(lang_code)
    TokensNoStop = None
    StopwordsRemoved = None
    if sw is not None:
        TokensNoStop = [t for t in Tokens if t.lower() not in sw]
        StopwordsRemoved = len(Tokens) - len(TokensNoStop)

    # NER – brak w NLTK
    Entities = []

    # Zbuduj wynik z DUŻYMI kluczami
    result = {
        "Tokens": Tokens,
        "Lemmas": Lemmas,
        "Entities": Entities,
        "Sentences": Sentences,
        "POS": POS,
        # Dodatkowe puste pola dla zgodności z app (jeśli używasz gdzieś indziej)
        "morph": [],
        "dependencies": [],
        "segment_words": Tokens,
    }

    # Dodaj stopwords TYLKO jeśli NLTK je ma lokalnie
    if TokensNoStop is not None and StopwordsRemoved is not None:
        result["TokensNoStop"] = TokensNoStop
        result["StopwordsRemoved"] = StopwordsRemoved

    return result
