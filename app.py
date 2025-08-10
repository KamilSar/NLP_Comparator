import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2

from libs import (
    spacy_wrapper,
    stanza_wrapper,
    transformers_wrapper,
    nltk_wrapper,
    textblob_wrapper,
    language_detector,
    benchmark,
    csv_writer,
)

# Wstępne załadowanie modeli transformers
transformers_wrapper.preload_models()

# Konfiguracja strony
st.set_page_config(page_title="Porównywarka NLP", layout="wide")
st.title("Porównywarka bibliotek NLP")

# === Wgrywanie pliku lub tekst ręczny ===
text = ""
uploaded_file = st.file_uploader("Wgraj plik TXT lub PDF", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

st.write("Lub wpisz tekst ręcznie:")
manual_input = st.text_area("Tekst (opcjonalnie)", height=200)

if manual_input.strip():
    text = manual_input.strip()

# === Jeśli jest tekst ===
if text:
    if len(text) > 10000:
        text = text[:10000]
        st.warning("Tekst został skrócony do 10 000 znaków.")

    lang = language_detector.detect_language(text)
    st.write(f"Wykryty język: {lang.upper()}")

    libraries = {
        "spaCy": lambda t: spacy_wrapper.process_text(t, lang),
        "Stanza": lambda t: stanza_wrapper.process_text(t, lang),
        "Transformers": lambda t: transformers_wrapper.process_text(t, lang),
        "TextBlob": textblob_wrapper.process_text,
        "NLTK": nltk_wrapper.process_text,
    }

    selected = st.multiselect("Wybierz biblioteki do analizy", libraries.keys(), default=list(libraries.keys()))

    results_data = []
    details = {}

    for name in selected:
        st.subheader(name)
        with st.spinner(f"Analizuję tekst ({name})..."):
            summary, results = benchmark.benchmark_nlp_library(name, libraries[name], text)

        results_data.append(summary)
        details[name] = results

        st.text(f"Czas wykonania: {summary['Execution Time (s)']} s")
        st.text(f"Tokeny: {summary['Tokens Count']}")
        st.text(f"Lematy: {summary['Lemmas Count']}")
        st.text(f"Byty NER: {summary['Entities Count']}")

        # Klucze z dużych liter
        tokens = results.get("Tokens", [])
        lemmas = results.get("Lemmas", [])
        entities = results.get("Entities", [])

        # Pokazanie pierwszych 20
        st.write("Tokeny (pierwsze 20):", tokens[:20])
        st.write("Lematy (pierwsze 20):", lemmas[:20])
        st.write("Byty NER (pierwsze 10):", entities[:10])

        # Pełne dane w rozwijanych sekcjach
        with st.expander("Tokeny (pełna lista)"):
            st.write(tokens)

        with st.expander("Lematy (pełna lista)"):
            st.write(lemmas)

        with st.expander("Byty NER (pełna lista)"):
            st.write(entities)

        if "morph" in results:
            st.write("Morfologia (pierwsze 10):")
            for token, morph in results["morph"][:10]:
                st.write(f"{token}: {morph}")

        if "dependencies" in results:
            st.write("Zależności składniowe (pierwsze 10):")
            for word, dep, head in results["dependencies"][:10]:
                st.write(f"{word} → {dep} ({head})")

        if "segment_words" in results:
            st.write("Segmentacja słów (pierwsze 20):", results["segment_words"][:20])

    # === Zapis i wykresy ===
    if results_data:
        csv_writer.save_results_to_csv(results_data, details)
        st.success("Wyniki zostały zapisane do pliku nlp_comparison_results.csv")

        df = pd.DataFrame(results_data)

        # Ustawienia ogólne do wykresów
        plot_kwargs = {
            "color": "gray",
            "edgecolor": "black"
        }
        font_title = 10
        font_label = 9
        font_ticks = 8

        st.subheader("Porównanie czasów wykonania")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.bar(df["Library"], df["Execution Time (s)"], **plot_kwargs)
        ax1.set_ylabel("Czas (s)", fontsize=font_label)
        ax1.set_title("Czas wykonania bibliotek", fontsize=font_title)
        ax1.tick_params(axis='x', labelsize=font_ticks)
        ax1.tick_params(axis='y', labelsize=font_ticks)
        st.pyplot(fig1)

        st.subheader("Porównanie liczby tokenów")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar(df["Library"], df["Tokens Count"], **plot_kwargs)
        ax2.set_ylabel("Liczba tokenów", fontsize=font_label)
        ax2.set_title("Tokenizacja", fontsize=font_title)
        ax2.tick_params(axis='x', labelsize=font_ticks)
        ax2.tick_params(axis='y', labelsize=font_ticks)
        st.pyplot(fig2)

        st.subheader("Porównanie liczby bytów NER")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.bar(df["Library"], df["Entities Count"], **plot_kwargs)
        ax3.set_ylabel("Liczba bytów NER", fontsize=font_label)
        ax3.set_title("Rozpoznawanie bytów nazwanych (NER)", fontsize=font_title)
        ax3.tick_params(axis='x', labelsize=font_ticks)
        ax3.tick_params(axis='y', labelsize=font_ticks)
        st.pyplot(fig3)
