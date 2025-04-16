import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch.nn as nn

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

class Seq2seqTrainer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, num_layers):
        super(Seq2seqTrainer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_dim, embed_dim)
        self.decoder_embedding = nn.Embedding(output_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, src, trg):
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)

        # Merge bidirectional outputs for decoder
        def concat_directions(h):
            return torch.cat((h[0:h.size(0):2], h[1:h.size(0):2]), dim=2)

        hidden = concat_directions(hidden)
        cell = concat_directions(cell)

        embedded_trg = self.decoder_embedding(trg)
        decoder_outputs, _ = self.decoder(embedded_trg, (hidden, cell))
        outputs = self.fc(decoder_outputs)
        return outputs

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