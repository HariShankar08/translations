import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

translate_languages = [
    "Assamese", "Bengali", "Bodo",
    "Dogri", "Gujarati", "Hindi",
    "Kannada", "Kashmiri", "Konkani",
    "Maithili", "Malayalam", "Manipuri",
    "Marathi", "Nepali", "Odia",
    "Punjabi", "Santali", "Sindhi",
    "Tamil", "Telugu", "Urdu"
]

tokenizer = AutoTokenizer.from_pretrained('sarvamai/sarvam-translate')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

model = AutoModelForCausalLM.from_pretrained('sarvamai/sarvam-translate', torch_dtype=dtype)

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

with open('en.jsonl') as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Translating", leave=False):
    import json
    obj = json.loads(line)
    for target_language in tqdm(translate_languages, desc="Iterating Languages"):
        question = translate_text(obj['question'], target_language)
        obj['question'] = question
        answer = translate_text(obj['answer'], target_language)
        obj['answer'] = answer
        with open('{}.jsonl'.format(target_language), 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(obj, ensure_ascii=False) + '\n')
