import json
import pandas as pd
from pathlib import Path
import os

# Define the directory containing the evaluation files
# Using the path from your previous context
eval_dir = Path("/home/rijulsaigal/NLP_Health/project/translations/Gemini_judgement")

def parse_evaluation_files(directory):
    data = []
    
    # Iterate through all json files matching the pattern
    # Pattern: {language}_{model}_answers_eval.json
    files = list(directory.glob("*_answers_eval.json"))
    
    if not files:
        print(f"No files found in {directory}")
        return pd.DataFrame()

    print(f"Found {len(files)} evaluation files. Processing...")

    for file_path in files:
        try:
            # 1. Parse Filename
            # Remove the suffix to get "{language}_{model}"
            stem = file_path.name.replace("_answers_eval.json", "")
            
            # Split into language and model
            # We assume the first underscore separates language from model
            # e.g., "Hindi_Llama2" -> ["Hindi", "Llama2"]
            if "_" in stem:
                parts = stem.split("_", 1)
                language = parts[0]
                model_name = parts[1]
            else:
                print(f"Skipping file with unexpected naming format: {file_path.name}")
                continue

            # 2. Read JSON content
            content = json.loads(file_path.read_text(encoding='utf-8'))
            outputs = content.get("Outputs", [])

            if not outputs:
                print(f"Warning: No outputs found in {file_path.name}")
                continue

            # 3. Calculate Averages for this file
            # Convert list of dicts to DataFrame for easy averaging
            df_temp = pd.DataFrame(outputs)
            
            # metrics to track
            metrics = ["Factual_consistency", "Coherence", "Similarity"]
            
            row = {
                "Model": model_name,
                "Language": language
            }
            
            for metric in metrics:
                if metric in df_temp.columns:
                    row[metric] = df_temp[metric].mean()
                else:
                    row[metric] = 0.0
            
            data.append(row)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    return pd.DataFrame(data)

def create_summary_table(df):
    if df.empty:
        return None

    # 1. Pivot the table to show metrics per language for each model
    # Index: Model
    # Columns: (Language, Metric) pairs
    pivot_df = df.pivot(index="Model", columns="Language", values=["Factual_consistency", "Coherence", "Similarity"])
    
    # Swap levels so Language is the top column header, then Metric
    # e.g., Hindi -> [Coherence, Factual_consistency, Similarity]
    pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # 2. Calculate "Average along averages" (Overall score per model across all languages)
    # We group by Model in the original flat dataframe and take the mean
    overall_stats = df.groupby("Model")[["Factual_consistency", "Coherence", "Similarity"]].mean()
    
    # Create a MultiIndex for these new columns so they can be joined
    # Top level: "Overall_Average", Second level: Metric name
    overall_stats.columns = pd.MultiIndex.from_product([["Overall_Average"], overall_stats.columns])

    # 3. Combine specific language scores with the overall average
    final_df = pd.concat([pivot_df, overall_stats], axis=1)
    
    return final_df

# --- Main Execution ---
if __name__ == "__main__":
    # Parse the data
    raw_df = parse_evaluation_files(eval_dir)
    
    if not raw_df.empty:
        # Generate the summary
        summary_table = create_summary_table(raw_df)
        
        # Display setup: show all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.2f}'.format)
        
        print("\n--- Evaluation Summary Table ---")
        print(summary_table)
        
        # Save to CSV
        output_csv_path = eval_dir / "evaluation_summary_matrix.csv"
        summary_table.to_csv(output_csv_path)
        print(f"\nSummary table saved to: {output_csv_path}")
        
        # Optional: Save a simplified 'Long' format version as well
        long_csv_path = eval_dir / "evaluation_summary_long.csv"
        raw_df.to_csv(long_csv_path, index=False)
        print(f"Raw averages (long format) saved to: {long_csv_path}")
    else:
        print("No data available to process.")