import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2
from collections import Counter  # do listy usuniętych stopwords

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

# (opcjonalne) preload transformers; inaczej pojawiaja sie komentarze później
try:
    transformers_wrapper.preload_models()
except Exception as e:
    print(f"Transformers preload warning: {e}")

st.set_page_config(page_title="Porównywarka NLP", layout="wide")
st.title("Porównywarka bibliotek NLP")

# Wgrywanie pliku lub tekst ręczny
text = ""
uploaded_file = st.file_uploader("Wgraj plik TXT lub PDF", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text

st.write("Lub wpisz tekst ręcznie:")
manual_input = st.text_area("Tekst (opcjonalnie)", height=200)
if manual_input.strip():
    text = manual_input.strip()

# analiza tekstu
if text:
    if len(text) > 10000:
        text = text[:10000]
        st.warning("Tekst został skrócony do 10 000 znaków.")

    lang = language_detector.detect_language(text)
    if lang not in ("pl", "en"):
        lang = "pl"
    st.write(f"Wykryty język: {lang.upper()}")

    # Biblioteki – uruchamiane TYLKO w wykrytym języku
    raw_libs = {
        "spaCy": spacy_wrapper.process_text,
        "Stanza": stanza_wrapper.process_text,
        "Transformers": transformers_wrapper.process_text,
        "TextBlob": textblob_wrapper.process_text,
        "NLTK": nltk_wrapper.process_text,
    }

    selected = st.multiselect(
        "Wybierz biblioteki do analizy",
        list(raw_libs.keys()),
        default=list(raw_libs.keys())
    )

    results_data = []
    details = {}

    for lib_name in selected:
        st.subheader(lib_name)
        wrapped = lambda t: raw_libs[lib_name](t, lang)

        with st.spinner(f"Analizuję tekst ({lib_name})..."):
            summary, results = benchmark.benchmark_nlp_library(lib_name, wrapped, text)

        results_data.append(summary)
        details[lib_name] = results

        # Odczyt
        tokens       = results.get("Tokens", [])
        lemmas       = results.get("Lemmas", [])
        entities     = results.get("Entities", [])
        sentences    = results.get("Sentences", [])
        pos_tags     = results.get("POS", [])

        # Pola stopwords- jeśli biblioteka posiada stopwords
        tokens_no_sw = results.get("TokensNoStop", [])
        removed_sw   = results.get("StopwordsRemoved", 0)
        has_sw_fields = ("TokensNoStop" in results) and ("StopwordsRemoved" in results)

        # METRYKI PODSTAWOWE
        st.text(f"Czas wykonania: {summary.get('Execution Time (s)', 0):.3f} s")
        st.text(f"Tokeny: {summary.get('Tokens Count', len(tokens))}")
        st.text(f"Lematy: {summary.get('Lemmas Count', len(lemmas))}")
        st.text(f"Byty NER: {summary.get('Entities Count', len(entities))}")

        # PODGLĄDY KRÓTKIE
        st.write("Tokeny (pierwsze 20):", tokens[:20])
        st.write("Lematy (pierwsze 20):", lemmas[:20])
        st.write("Byty NER (pierwsze 10):", entities[:10])

        #  Segemntacja zdan
        st.markdown("### Segmentacja zdań (SBD)")
        st.write(f"Liczba zdań: {len(sentences)}")
        if len(sentences) > 0:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            st.write(f"Średnia długość zdania (w tokenach): {avg_len:.2f}")
            st.write("Pierwsze 3 zdania:")
            for i, s in enumerate(sentences[:3], start=1):
                st.write(f"{i}. {s}")
        else:
            st.write([])

        # POS- tagging
        st.markdown("### POS-tagging (pierwsze 20)")
        st.write(pos_tags[:20] if pos_tags else [])

        # Morfologia, dependencies, segmentacja słów
        if "morph" in results:
            st.write("Morfologia (pierwsze 10):", results.get("morph", [])[:10])
        if "dependencies" in results:
            st.write("Zależności składniowe (pierwsze 10):", results.get("dependencies", [])[:10])
        if "segment_words" in results:
            st.write("Segmentacja słów (pierwsze 20):", results.get("segment_words", [])[:20])

        # Pełne listy na końcu
        with st.expander("Tokeny (pełna lista)"):
            st.write(tokens)

        with st.expander("Lematy (pełna lista)"):
            st.write(lemmas)

        with st.expander("Byty NER (pełna lista)"):
            st.write(entities)

        # Stopwords (statystyki + pełna lista usuniętych)
        removed_list = []
        if tokens and tokens_no_sw:
            # policz usunięte jako różnicę multizbiorów (Tokens - TokensNoStop)
            cntr_after = Counter([t.lower() for t in tokens_no_sw])
            for tok in tokens:
                ltok = tok.lower()
                if cntr_after.get(ltok, 0) > 0:
                    cntr_after[ltok] -= 1
                else:
                    removed_list.append(tok)

        with st.expander("Stopwords (statystyki + pełna lista usuniętych)"):
            st.write(f"Liczba tokenów PRZED: {len(tokens)}")
            # Jeśli biblioteka nie ma stopwords, te wartości będą 0/0/[]
            st.write(f"Liczba tokenów PO: {len(tokens_no_sw) if has_sw_fields else 0}")
            st.write(f"Usunięte stopwords: {removed_sw if has_sw_fields else 0}")
            st.write("Usunięte stopwords (pełna lista):", removed_list)

    # Zapis i wykresy
    if results_data:
        csv_writer.save_results_to_csv(results_data, details)
        st.success("Wyniki zostały zapisane do pliku nlp_comparison_results.csv")

        df = pd.DataFrame(results_data)
        plot_kwargs = {"color": "gray", "edgecolor": "black"}
        font_title = 10
        font_label = 9
        font_ticks = 8

        st.subheader("Porównanie czasów wykonania")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.bar(df["Library"], df["Execution Time (s)"], **plot_kwargs)
        ax1.set_ylabel("Czas (s)", fontsize=font_label)
        st.pyplot(fig1)

        st.subheader("Porównanie liczby tokenów")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar(df["Library"], df["Tokens Count"], **plot_kwargs)
        ax2.set_ylabel("Liczba tokenów", fontsize=font_label)
        st.pyplot(fig2)

        st.subheader("Porównanie liczby bytów NER")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.bar(df["Library"], df["Entities Count"], **plot_kwargs)
        ax3.set_ylabel("Liczba bytów NER", fontsize=font_label)
        st.pyplot(fig3)
