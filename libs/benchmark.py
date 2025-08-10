# libs/benchmark.py
import psutil
import os
import time

process = psutil.Process(os.getpid())

def _get_list(results, big, small):
    """Pomocniczo: preferuj DUŻY klucz, w ostateczności mały."""
    return results.get(big, results.get(small, []))

def _get_val(results, big, small, default=None):
    return results.get(big, results.get(small, default))

def benchmark_nlp_library(name, func, text):
    # pamiętaj: func w app.py jest owinięte tak, by przekazać lang_code
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_cpu = psutil.cpu_percent(interval=1)

    start_time = time.time()
    results = func(text)               # <- pełne wyniki z wrappera
    end_time = time.time()

    end_mem = process.memory_info().rss / (1024 * 1024)
    end_cpu = psutil.cpu_percent(interval=1)

    duration = round(end_time - start_time, 3)
    memory_diff = round(end_mem - start_mem, 3)

    # --- Pobieranie danych (DUŻE klucze z fallbackiem) ---
    tokens     = _get_list(results, "Tokens", "tokens")
    lemmas     = _get_list(results, "Lemmas", "lemmas")
    entities   = _get_list(results, "Entities", "entities")
    sentences  = _get_list(results, "Sentences", "sentences")
    pos_tags   = _get_list(results, "POS", "pos_tags")

    # stopwords – tylko jeśli biblioteka natywnie zwraca
    tokens_no_sw     = _get_list(results, "TokensNoStop", "tokens_no_stopwords")
    stopwords_removed = _get_val(results, "StopwordsRemoved", "stopwords_removed_count", 0)

    # --- Logi terminalowe (pomocnicze) ---
    print(f"⏱️ Czas wykonania: {duration}s | ΔRAM: {memory_diff} MB | CPU(before/after) {start_cpu}% → {end_cpu}%")
    print(f"📌 Tokeny(20): {tokens[:20]}")
    print(f"📌 Lematy(20): {lemmas[:20]}")
    print(f"📌 NER(10): {entities[:10]}")
    if sentences:
        print(f"📌 Zdania: {len(sentences)}; pierwsze: {sentences[:2]}")
    if pos_tags:
        print(f"📌 POS(10): {pos_tags[:10]}")
    if "TokensNoStop" in results or "tokens_no_stopwords" in results:
        print(f"📌 Stopwords: usunięto {stopwords_removed}; po filtracji: {len(tokens_no_sw)} tokenów")
    print()

    # --- Podsumowanie do CSV/wykresów ---
    summary = {
        "Library": name,
        "Execution Time (s)": duration,
        "Memory Increase (MB)": memory_diff,
        "CPU Before (%)": start_cpu,
        "CPU After (%)": end_cpu,

        "Tokens Count": len(tokens),
        "Lemmas Count": len(lemmas),
        "Entities Count": len(entities),

        # dodatkowe, ale neutralne gdy puste
        "Sentences Count": len(sentences),
        "Has POS": bool(pos_tags),
        "Has Stopwords": ("TokensNoStop" in results) or ("tokens_no_stopwords" in results),
        "Tokens After Stopwords": len(tokens_no_sw) if tokens_no_sw else 0,
        "Stopwords Removed": int(stopwords_removed) if stopwords_removed else 0,
    }

    # Zwracamy pełne results, nie „odchudzone” – app.py ma wtedy wszystko
    return summary, results
