import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Load Dataset
df = pd.read_csv("english_french.csv").dropna()
df = df[['English', 'French']].rename(columns={"English": "en", "French": "fr"})

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load Tokenizer & Model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocessing
def preprocess(batch):
    inputs = tokenizer(batch["en"], padding="max_length", truncation=True, max_length=50)
    targets = tokenizer(batch["fr"], padding="max_length", truncation=True, max_length=50)
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=["en", "fr"])

# Split Dataset into train and test
dataset = dataset.train_test_split(test_size=0.1)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # You can set to True if you have a compatible GPU and want mixed precision training
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train Model
trainer.train()

# Save Model and Tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")