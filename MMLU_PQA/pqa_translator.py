import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

translate_languages = [
    "Assamese", "Bengali", "Bodo", 
    "Dogri", "Gujarati", "Hindi", 
    "Kannada", "Kashmiri", "Konkani", 
    "Maithili", "Malayalam", "Manipuri", 
    "Marathi", "Nepali", "Odia", 
    "Punjabi", "Santali", "Sindhi", 
    "Tamil", "Telugu", "Urdu"
]

mmlu_subsets = [
    'pqa_unlabeled'
]

tokenizer = AutoTokenizer.from_pretrained('sarvamai/sarvam-translate')

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

model = AutoModelForCausalLM.from_pretrained('sarvamai/sarvam-translate', torch_dtype=dtype).to(device)

def translate_text(text, target_language, speaker_gender='male'):
    messages = [
        {"role": "system", "content": "Translate the text below to {}, as a {} speaker.".format(target_language, speaker_gender)},
        {"role": "user", "content": text}
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.01,
        num_return_sequences=1
    )

    output_ids = generated_ids[0][model_inputs['input_ids'].shape[1]:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text.strip()

# Iterate over all the MMLU subsets.
for subset in mmlu_subsets:
    # Load the dataset for the current subset.
    dataset = load_dataset('qiaojin/PubMedQA', subset, streaming=True)

    # Iterate over all the splits.
    for split in dataset:
        # Iterate over the examples in the split.
        for example in tqdm(dataset[split], desc=f"Translating {subset} - {split}", leave=False):
            # the `example` contains the question, choices, and a label (ABCD) representing the correct answer
            #  Extract the correct answer from the choices.
            # Translate the question and choices to each target language, both as male and female.
            question = example['question']
            # choices = example['choices']
            # answer = example['answer']
            correct_answer_text = example['long_answer']

            for language in tqdm(translate_languages, desc="Translating example"):
                tq_set = set()
                ta_set = set()
                for gender in ['male']:
                    translated_question = (translate_text(question, language, gender))
                    translated_answer = (translate_text(correct_answer_text, language, gender))
                    tq_set.add(translated_question)
                    ta_set.add(translated_answer)
                
                for tq in tq_set:
                    for ta in ta_set:
                        new_example = {
                            'question': tq,
                            'answer': ta,
                            'original_question': question,
                            'original_answer': correct_answer_text,
                            'language': language
                        }
                        # Save the new example to a file.
                        output_file = f'pqa_{subset}.jsonl'
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(new_example, ensure_ascii=False) + '\n')
