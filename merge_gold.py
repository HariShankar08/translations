import json
import os

# --- Configuration ---
GOLD_DIR = 'Gold'
OUTPUTS_DIR = 'outputs'
NEW_DIR = 'outputs_with_gold'
# ---------------------

def process_files():
    """
    Finds gold files, iterates through matching output files, 
    and adds the gold answer to each line.
    """
    
    # Create the new directory if it doesn't exist
    os.makedirs(NEW_DIR, exist_ok=True)
    print(f"Created new directory: {NEW_DIR}")

    # 1. Find all gold files
    try:
        gold_files = [f for f in os.listdir(GOLD_DIR) if f.endswith('.jsonl')]
    except FileNotFoundError:
        print(f"Error: The '{GOLD_DIR}' directory was not found.")
        print("Please create the 'Gold' directory and add your gold files.")
        return
    
    if not gold_files:
        print(f"No .jsonl files found in {GOLD_DIR}")
        return

    # 2. Loop through each gold file
    for gold_file_name in gold_files:
        base_name = gold_file_name.replace('.jsonl', '')
        gold_file_path = os.path.join(GOLD_DIR, gold_file_name)
        
        print(f"\n--- Processing for base: {base_name} ---")

        # 3. Read all gold answers into memory
        gold_answers = []
        try:
            with open(gold_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        gold_data = json.loads(line)
                        gold_answers.append(gold_data.get('answer'))
                    except json.JSONDecodeError:
                        print(f"  Warning: Skipping malformed line {line_num+1} in {gold_file_name}")
                        gold_answers.append(None) # Add placeholder to keep line counts in sync
        except FileNotFoundError:
            print(f"  Error: Could not open {gold_file_path}. Skipping.")
            continue

        print(f"  Loaded {len(gold_answers)} answers from {gold_file_name}")

        # 4. Find all matching files in the outputs directory
        try:
            output_files = [f for f in os.listdir(OUTPUTS_DIR) 
                            if f.startswith(base_name) and f.endswith('.jsonl')]
        except FileNotFoundError:
            print(f"Error: The '{OUTPUTS_DIR}' directory was not found.")
            print("Please create the 'outputs' directory and add your model output files.")
            continue
            
        if not output_files:
            print(f"  No matching files found in {OUTPUTS_DIR} starting with '{base_name}'")
            continue

        # 5. Process each matching output file
        for output_file_name in output_files:
            output_file_path = os.path.join(OUTPUTS_DIR, output_file_name)
            new_output_file_path = os.path.join(NEW_DIR, output_file_name)
            
            print(f"  Processing: {output_file_name}")
            
            line_index = 0
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f_in, \
                     open(new_output_file_path, 'w', encoding='utf-8') as f_out:
                    
                    for model_line_raw in f_in:
                        # Check if we have a gold answer for this line
                        if line_index >= len(gold_answers):
                            print(f"    Warning: {output_file_name} has more lines than {gold_file_name}. Truncating.")
                            break
                        
                        try:
                            # Load the model's data
                            model_data = json.loads(model_line_raw)
                            
                            # Add the gold answer
                            model_data['gold_answer'] = gold_answers[line_index]
                            
                            # Write the new, combined line to the new file
                            f_out.write(str(line_index)+","+json.dumps(model_data) + '\n')
                            
                        except json.JSONDecodeError:
                            print(f"    Warning: Skipping malformed line {line_index+1} in {output_file_name}")
                        
                        line_index += 1
                
                print(f"    Success: Created {new_output_file_path}")

            except Exception as e:
                print(f"    Error processing {output_file_name}: {e}")

    print("\n--- All processing complete. ---")

# --- Run the script ---
if __name__ == "__main__":
    # Before running, make sure your directory structure is:
    # .
    # ├── this_script.py
    # ├── Gold/
    # │   └── Malayalam.jsonl
    # │   └── (other_gold_files.jsonl...)
    # └── outputs/
    #     └── Malayalam_google_gemma-2-2b-it_answers.jsonl
    #     └── Malayalam_google_gemma-3-4b-it_answers.jsonl
    #     └── (other_output_files.jsonl...)
    
    process_files()