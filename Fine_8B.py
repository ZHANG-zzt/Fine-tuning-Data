import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
import os

# Define model name 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_name = "/data1/zzt/models/deepseek-r1"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'  # # Ensure the padding_side keep 'right'



# Load the model 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"":0},  
    trust_remote_code=True  
)

# LoRA Configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # Fine-tuning model as autoregressive model
    r=16,                   # Rank for LoRA low-rank decomposition
    lora_alpha=32,         # LoRA scaling factor
    target_modules=["q_proj", "v_proj"],  # Target modules based on Qwen2 architecture
    lora_dropout=0.05,     # Dropout probability
    bias="none",           # Do not train bias parameters
    init_lora_weights=True,  # Initialize LoRA layer weights
    inference_mode=False   # Enable training mode
)


model = get_peft_model(model, lora_config)

# Define parameters
training_arguments = TrainingArguments(
    output_dir="/data1/zzt/SLM/Llama_8B/Llama_8B_2025",
    eval_strategy="no",   # Disable evaluation
    optim="paged_adamw_8bit",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=5,
    learning_rate=1e-4,
    fp16=False,  # Select based on hardware support
    bf16=False,  
    num_train_epochs=6,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
)

# Process Data
def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    blue_float = [20, 10, 0]
    red_float = [-10, 20, 0]
    
    # Use only input to construct the prompt
    input_text = f"User: {example['input']}\n\n"
    output_text = example['output']
    output_text = output_text.replace("{blue_float}", str(blue_float))  # Replace blue_float
    output_text = output_text.replace("{red_float}", str(red_float))    # Replace red_floa
    response_text = f"Assistant: {example['output']}{tokenizer.eos_token}"

    # Tokenize input and response
    input_encoding = tokenizer(input_text, add_special_tokens=False)
    response_encoding = tokenizer(response_text, add_special_tokens=False)

    input_ids = input_encoding["input_ids"] + response_encoding["input_ids"]
    attention_mask = input_encoding["attention_mask"] + response_encoding["attention_mask"]
    labels = [-100] * len(input_encoding["input_ids"]) + response_encoding["input_ids"]

    # Apply truncation if the total length exceeds MAX_LENGTH
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# Load the training datasets
df = pd.read_json('/data1/zzt/SLM/datasets_revise/generated_trajectory_data2.json') 
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# Creating the trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# Using the model 
def generate_response(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, 
    max_length=max_length,
    no_repeat_ngram_size=2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    
# Response Templete 
def answer_template(prompt):
    return f"User: {prompt}\n\nAssistant:"

# Example  question 
question = "Set the trajectory with the speeds along the X and Z axes set to 0.95 m/s and -0.21 m/s respectively, while the Y-coordinate follows a sine wave with a period of 21 seconds and an amplitude of -6.24 meters."
print("Answer before training:")
print(generate_response(model, tokenizer, answer_template(question)))

# Start train 
trainer.train()
# Save the model
trainer.save_model("/data1/zzt/SLM/Llama_8B/Llama_8b_LoRA")

# Use the trained model to answer questions
def generate_response(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# question = "Set the trajectory with the speeds along the X and Z axes set to 1.95 m/s and -0.46 m/s respectively, while the Y-coordinate follows a sine wave with a period of 17 seconds and an amplitude of -3.24 meters."

question = "Set the weight matrix for control input to diag(0.55, 0.55, 0.55, 0.55, 0.55, 0.55)."
print("Answer after training:")
print(generate_response(model, tokenizer, answer_template(question)))

