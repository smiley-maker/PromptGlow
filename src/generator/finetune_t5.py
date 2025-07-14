import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# ==== Step 1: Load and Prepare Data ====
print("Loading dataset...")
df = pd.read_csv("data/splits/train.csv")  # columns: short_prompt, long_prompt
val_df = pd.read_csv("data/splits/val.csv")

# Create Hugging Face Dataset
dataset = Dataset.from_pandas(df)
val_dataset = Dataset.from_pandas(val_df)
#dataset = dataset.train_test_split(test_size=0.1)

# ==== Step 2: Load Base Model and Tokenizer ====
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ==== Step 3: Apply LoRA ====
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
#    target_modules=["q", "v"]  # Applies LoRA to attention projections
)

model = get_peft_model(model, peft_config)

# ==== Step 4: Tokenize Data ====
max_length = 64

def tokenize_function(example):
    input_enc = tokenizer(example["short_prompt"], truncation=True, padding="max_length", max_length=max_length)
    target_enc = tokenizer(example["long_prompt"], truncation=True, padding="max_length", max_length=max_length)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# ==== Step 5: Define Training Arguments ====
training_args = TrainingArguments(
    output_dir="model_output/lora-flan-t5",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    logging_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    learning_rate=2e-4,
    weight_decay=0.01,
    report_to="none",
    use_mps_device=True
)

# ==== Step 6: Trainer ====
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ==== Step 7: Train ====
print("Starting LoRA fine-tuning...")
trainer.train()

# ==== Step 8: Save Adapter ====
print("Saving LoRA adapter...")
model.save_pretrained("model_output/lora-flan-t5")
tokenizer.save_pretrained("model_output/lora-flan-t5")
print("Done!")
