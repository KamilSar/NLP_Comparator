import pandas as pd

def save_results_to_csv(results_data, detailed_output, output_file="nlp_comparison_results.csv"):
    df = pd.DataFrame(results_data)
    df = pd.concat([df, pd.DataFrame([{}])], ignore_index=True)

    for lib_name, data in detailed_output.items():
        df = pd.concat([
            df,
            pd.DataFrame([
                {"Library": lib_name + " - Tokens", "Details": ", ".join(data.get("Tokens", [])[:50]) + " ..."},
                {"Library": lib_name + " - Lemmas", "Details": ", ".join(data.get("Lemmas", [])[:50]) + " ..."},
                {"Library": lib_name + " - Entities", "Details": str(data.get("Entities", []))},
                {}
            ])
        ], ignore_index=True)

    df.to_csv(output_file, index=False)
    print(f"📄 Wyniki zapisane do: {output_file}")
