# libs/benchmark.py
import psutil
import os
import time

process = psutil.Process(os.getpid())

def benchmark_nlp_library(name, func, text):
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_cpu = psutil.cpu_percent(interval=1)

    start_time = time.time()
    results = func(text)
    end_time = time.time()

    end_mem = process.memory_info().rss / (1024 * 1024)
    end_cpu = psutil.cpu_percent(interval=1)

    duration = round(end_time - start_time, 3)
    memory_diff = round(end_mem - start_mem, 3)

    # Terminal output
    print(f"⏱️ Czas wykonania: {duration}s")
    print(f"📌 Tokeny: {results.get('tokens', [])[:20]} ...")
    print(f"📌 Lematy: {results.get('lemmas', [])[:20]} ...")
    print(f"📌 NER: {results.get('entities', [])[:10]}\n")

    if "morph" in results:
        print("🧬 Morfologia (pierwsze 10):")
        for token, morph in results["morph"][:10]:
            print(f"  {token}: {morph}")
        print()

    if "dependencies" in results:
        print("🧠 Zależności składniowe (pierwsze 10):")
        for word, dep, head in results["dependencies"][:10]:
            print(f"  {word} → {dep} ({head})")
        print()

    if "segment_words" in results:
        print(f"🪓 Segmentacja słów (pierwsze 20): {results['segment_words'][:20]}")
        print()

    summary = {
        "Library": name,
        "Execution Time (s)": duration,
        "Memory Increase (MB)": memory_diff,
        "CPU Before (%)": start_cpu,
        "CPU After (%)": end_cpu,
        "Tokens Count": len(results.get("tokens", [])),
        "Lemmas Count": len(results.get("lemmas", [])),
        "Entities Count": len(results.get("entities", [])),
        "Has Morphology": "morph" in results,
        "Has Dependencies": "dependencies" in results,
        "Has Word Segmentation": "segment_words" in results,
    }

    detailed = {
        "Tokens": results.get("tokens", []),
        "Lemmas": results.get("lemmas", []),
        "Entities": results.get("entities", []),
    }

    return summary, detailed
