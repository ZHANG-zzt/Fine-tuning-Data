import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re

# Model Initialization
model_name = "/data1/zzt/models/deepseek-r1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


model = AutoModelForCausalLM.from_pretrained(
    "/data1/zzt/SLM/Llama_8B/Llama_8b_LoRA",
    device_map="auto",
    trust_remote_code=True
)

# Generate Response Function
def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        #do_sample=False,
        do_sample=True,      # Enable sampling 
        temperature=0.7,     # Setting these parameters 
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.replace(prompt, "").strip()
    response = response.split("User:")[0].strip()
    return response

# Clean function to normalize the text (ignores punctuation, spaces, and case)
def clean_text(text):
    # Remove punctuation, convert to lowercase, and strip extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()  # Convert to lowercase and strip extra spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one space
    return text

# Prompt Template
def answer_template(prompt):
    return f"User: {prompt}\n\nAssistant:"


test_df = pd.read_csv("test_questions.csv")

correct = 0
total_completion_tokens = 0
results = []

# Evaluate model accuracy over 100 questions
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    question = row["question"]
    expected_answer = row["expected_answer"].strip()

    prompt = answer_template(question)
    model_answer = generate_response(model, tokenizer, prompt).strip()
    print(f"Model Answer: {model_answer}\n")
    
    completion_tokens = len(tokenizer.encode(model_answer))
    total_completion_tokens += completion_tokens

    # Clean both model's answer and expected answer
    cleaned_model_answer = clean_text(model_answer)
    cleaned_expected_answer = clean_text(expected_answer)
    
    # Compare cleaned answers
    is_correct = (cleaned_model_answer == cleaned_expected_answer)
    correct += int(is_correct)

    results.append({
        "question": question,
        "expected": expected_answer,
        "model_answer": model_answer,
        "correct": is_correct
    })

results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)

accuracy = correct / len(test_df) * 100
print(f"{accuracy:.2f}%")

avg_completion_tokens = total_completion_tokens / len(test_df)
print(f"{avg_completion_tokens:.2f}")
