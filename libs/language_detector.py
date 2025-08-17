from langdetect import detect

def detect_language(text: str) -> str:
    """
    Wykrywa język tekstu na podstawie pierwszych 500 znaków
    i wypisuje wynik w terminalu.

    Zwraca kod języka ISO 639-1-  'pl', 'en'.
    """
    try:
        lang_code = detect(text[:500])
        print(f"🧭 Wykryty język: {lang_code.upper()}")
        return lang_code
    except Exception as e:
        print(f" Nie udało się wykryć języka: {e}")
        return "unknown"
