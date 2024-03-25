import torch
import pandas as pd
from tqdm.notebook import tqdm
import gc

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
from huggingface_hub import login

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# SET pre_training TO False FOR FINE-TUNING
pre_training = True
# SET data_path TO YOUR DATASET FILE (csv)
data_path = "PATH TO DATA"
# ENTER YOUR HUGGINFACE TOKEN TO PUSH THE MODEL TO HF HUB AND ACCESS GEMMA
hf_token = "ENTER YOUR HUGGINGFACE TOKEN"

login(token = hf_token)

if pre_training:
    model_id = "google/gemma-2b-it"
else:
    model_id = "merged_model_curricular"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, padding_side='left')

def get_completion(query_list, model, tokenizer) -> str:
    prompt_template = """
    <start_of_turn>user
    {query}
    <end_of_turn>\n<start_of_turn>model
    """
    prompt_list = []
    for query in query_list:
        prompt = prompt_template.format(query=query)
        prompt_list.append(prompt)

    encodeds = tokenizer(prompt_list, return_tensors="pt", padding='max_length', max_length=2048, add_special_tokens=True)

    model_inputs = encodeds.to(device)


    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id, num_beams=3, num_return_sequences=1)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded


def generate_prompt_diagnosis(data_point):
    prefix = "<start_of_turn>user Act as a psychologist and give a diagnosis for the text in backticks. Include the main theme involved, any sub themes observed, predominant emotion, other observed emotions, severity of negative emotion, risk of suicide, diagnosis of mental health syndrome and any environmental stressors. Keep in mind that any of the above can be missing, report None in that case. Do not talk about anything more. Answer in a single paragraph."
    return prefix + "```" + data_point['TEXT'] + "```<end_of_turn>\n<start_of_turn>model " + data_point['Generated Diagnosis Summary'] + " <end_of_turn>"

def generate_prompt_curricular(data_point):
    prefix = "<start_of_turn>user You are a student studying psychology. Given the following text from a curricular in backticks, try to learn and understand it and predict the next sentences."
    words = data_point['post'].split(" ")
    first_two_thirds = " ".join(words[:2*len(words)//3])
    final_third = " ".join(words[2*len(words)//3:])
    return prefix + "```" + first_two_thirds + "```<end_of_turn>\n<start_of_turn>model " + final_third + " <end_of_turn>"

if pre_training:
    df = pd.read_csv(data_path)
    prompt_column = [generate_prompt_curricular(row) for i, row in df.iterrows()]
    df['prompt'] = prompt_column
    data = Dataset.from_pandas(df)
else:
    df = pd.read_csv(data_path, delimiter='|')
    prompt_column = [generate_prompt_diagnosis(row) for i, row in df.iterrows()]
    df['prompt'] = prompt_column
    data = Dataset.from_pandas(df)


data = data.shuffle(seed=1234)  # Shuffle data here
data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

data = data.train_test_split(test_size=0.2)
train_dataset = data["train"]
test_dataset = data["test"]


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
print("Modules being trained: ", modules)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="prompt",
    max_seq_length=2048,
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        warmup_steps=0.03,
        num_train_epochs=1,
        # max_steps=30,
        learning_rate=1e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()

if pre_training:
    new_model = "merged_model_curricular"
else:
    new_model = "merged_model_CASE"
    
torch.cuda.empty_cache()

trainer.save_model(new_model)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
if pre_training:
    merged_model.save_pretrained("merged_model_curricular", safe_serialization=True)
    tokenizer.save_pretrained("merged_model_curricular")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
else:
    merged_model.save_pretrained("merged_model_CASE", safe_serialization=True)
    tokenizer.save_pretrained("merged_model_CASE")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

# Generating diagnosis

if not pre_training:
    df = test_dataset.to_pandas()
    col_to_use = 'TEXT'

    torch.cuda.empty_cache()

    prompt_prefix = "Act as a professional psychologist and give a psychological breakdown for the text in backticks. This text is a person acting for advice on a mental health forum, hence do not misinterpret their questioning nature as being existential or suicidal and do not be overly sensistive to the same. Include the main theme involved, any sub themes, predominant emotion, any other emotions expressed, positive emotions if any, severity of negative emotion, any risk of suicide or self harm ONLY IF IT IS VERY VERY CLEARLY STATED IN THE TEXT, diagnosis of a mental health syndrome if any, and any environmental stressors if present. Take very special care not to overestimate the risk of suicide and self harm, if you say that the person is suicidal, give very convincing reason for the same. Keep in mind that any of the above can be missing, report None in that case. Give reasons behind every conclusion that you make based on the text and your psychology knowledge. DO NOT be very oversensitive towards syndromes like depression, anxiety and suicide. Do not give any recommendations.```"

    import warnings
    warnings.filterwarnings("ignore")

    batch_size = 2
    def generate_diagnosis_given_text(df):
        diagnosis_list = []
        progress_bar = tqdm(range(df.shape[0]//batch_size))
        query_list = []
        for i, row in df.iterrows():
            if ((i + 1) % batch_size == 0) or (i == df.shape[0] - 1):
                query_list.append(prompt_prefix + row[col_to_use] + "```")

                result = get_completion(query_list, model=model, tokenizer=tokenizer)
                query_list = []
                diagnosis_list.extend([diag.split("model")[1].replace("\n", ". ").replace("*", "")[8:] for diag in result])
                # diagnosis_list.append(result.split("model")[1].replace("\n", ". ").replace("*", ""))
                progress_bar.update()
                torch.cuda.empty_cache()
                gc.collect()
                # break
            else:
                query_list.append(prompt_prefix + row[col_to_use] + "```")
                
        return diagnosis_list

    import warnings
    warnings.filterwarnings("ignore")

    torch.cuda.empty_cache()
    gc.collect()
    diag_list = generate_diagnosis_given_text(df)

    len(diag_list) == df.shape[0]

    df['Predicted Diagnosis'] = diag_list

    df.to_csv('DataWithPredictions.csv', index=False, sep='|')