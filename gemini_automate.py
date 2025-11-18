from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic import BaseModel, Field
from pathlib import Path
from time import sleep
import os

# Define paths
curpath = Path("/home/rijulsaigal/NLP_Health/project/translations/outputs_with_gold")
savepath = Path("/home/rijulsaigal/NLP_Health/project/translations/Gemini_judgement")

# Ensure output directory exists
savepath.mkdir(parents=True, exist_ok=True)

# Setup Provider and Model
# NOTE: It is best practice to use environment variables for API keys. 
# If you must hardcode, ensure this file is not shared publicly.
api_key = "AIzaSyD9ksu1uRZkRhw9GcioTTqtGXbwwMuKSPI" 
provider = GoogleProvider(api_key=api_key)
model = GoogleModel('gemini-2.5-flash-lite', provider=provider)

# Define Data Models
class Output(BaseModel):
    Line_number: int = Field(description="Line number from file")
    # Removed text fields to save tokens as per your latest snippet
    Factual_consistency: int = Field(description="factual consistency(scored from 1-10) between answer and discharge summary")
    Coherence: int = Field(description="coherence of the answer(scored from 1-10)")
    Similarity: int = Field(description="similarity between answer and gold answer(scored from 1-10)")

class Output_lines(BaseModel):
    Outputs: list[Output] 

# Initialize Agent
agent = Agent(model, output_type=Output_lines)

# Batch size configuration
BATCH_SIZE = 15

# Helper function to chunk list
def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

# System Prompt
system_prompt = """You are evaluating question answer pairs for medical QA models.
Score answers based on factual consistency with the discharge summary(1-10), overall coherence(1-10), and similarity to gold answer(1-10).
For the provided lines, output scores for each of these metrics for each line."""

# Main Processing Loop
# 1. Sort the files
files = sorted(curpath.glob("*.jsonl"))

for path in files:
    # 2. Check if output file already exists
    output_file = savepath / (path.stem + "_eval.json")
    
    if output_file.exists():
        print(f"Skipping {path.name} - Output already exists at {output_file.name}")
        continue

    print(f"Processing file: {path.name}")
    
    try:
        # Read all lines from the file
        all_lines = path.read_text(encoding='utf-8').strip().split('\n')
        
        # List to hold all aggregated results for this specific file
        file_aggregated_results = []
        
        # Process in chunks
        for chunk_index, batch in enumerate(chunk_list(all_lines, BATCH_SIZE)):
            print(f"  - Processing batch {chunk_index + 1}...")
            
            # Join the batch back into a string to send to the model
            batch_content = "\n".join(batch)
            
            try:
                # Run the agent on just this batch
                result = agent.run_sync(
                    [
                        system_prompt,
                        batch_content # Passing text string directly
                    ]
                )
                
                # Print usage stats
                print(f"    Usage: {result.usage().output_tokens} output tokens, {result.usage().input_tokens} input tokens")

                # Append the results from this batch to the main list
                # Note: Using .data.Outputs which is standard for pydantic-ai results
                file_aggregated_results.extend(result.output.Outputs)
                
                # Sleep between batches to respect rate limits (RPM and TPM)
                sleep(10) 
                
            except Exception as e:
                print(f"Error processing batch {chunk_index + 1} in {path.name}: {e}")
                # Optionally continue or break based on preference
                continue

        # Create the final object containing all lines for this file
        final_output = Output_lines(Outputs=file_aggregated_results)
        
        # Write the complete aggregated result to the JSON file
        output_file.write_text(final_output.model_dump_json(indent=4))
        print(f"Finished {path.name}. Saved to {output_file}")

    except Exception as e:
        print(f"Failed to read or process file {path.name}: {e}")

print("All files processed.")