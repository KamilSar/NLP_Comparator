from libs import (
    spacy_wrapper,
    nltk_wrapper,
    stanza_wrapper,
    transformers_wrapper,
    textblob_wrapper,
    txt_reader,
    language_detector,
    benchmark,
    csv_writer,
)

# Wstępne ładowanie modeli Transformers
transformers_wrapper.preload_models()

# Pobierz tekst od użytkownika
text = txt_reader.get_text_interactive()

if not text.strip():
    print(" Brak tekstu do analizy. Zakończono.")
    exit(1)

# Skracanie tekstu jeśli za długi
if len(text) > 10000:
    text = text[:10000]
    print(" Tekst został skrócony do 10 000 znaków.\n")

# Detekcja języka
lang = language_detector.detect_language(text)

# Lista bibliotek do porównania
libraries = [
    ("spaCy", lambda t: spacy_wrapper.process_text(t, lang)),
    ("Stanza", lambda t: stanza_wrapper.process_text(t, lang)),
    ("Transformers", lambda t: transformers_wrapper.process_text(t, lang)),
    ("TextBlob", textblob_wrapper.process_text),
    ("NLTK", nltk_wrapper.process_text),
]

# Benchmarkowanie i analiza wyników
results_data = []
details = {}

print(" Rozpoczynam analizę NLP...\n")

for name, func in libraries:
    print(f"=== [{name}]")
    summary, detail = benchmark.benchmark_nlp_library(name, func, text)
    results_data.append(summary)
    details[name] = detail

# Eksport wyników do CSV
csv_writer.save_results_to_csv(results_data, details)
