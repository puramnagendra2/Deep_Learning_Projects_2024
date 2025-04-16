```python
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
```

### Overview of the Project

1. **Objective of the Project**  
    The project focuses on building a machine translation system that translates English text into French using a pre-trained MarianMT model from the Hugging Face Transformers library.

2. **Dataset Preparation**  
    A dataset containing English and French sentence pairs is preprocessed. It is converted into a Hugging Face Dataset format, tokenized, and split into training and evaluation subsets for model training.

3. **Model and Tokenizer**  
    The project utilizes the "Helsinki-NLP/opus-mt-en-fr" MarianMT model and tokenizer. These components are pre-trained for English-to-French translation, ensuring a strong starting point for fine-tuning.

4. **Training Process**  
    The Seq2SeqTrainer class is employed to fine-tune the MarianMT model. Training arguments, such as learning rate, batch size, and number of epochs, are configured to optimize the model's performance.

5. **Model Saving and Reusability**  
    After training, the fine-tuned model and tokenizer are saved locally. This allows for easy reuse of the trained model for future translation tasks without retraining.

### Problem Statement
1. **Language Barrier**  
    Effective communication between English and French speakers is often hindered by the lack of accurate and accessible translation tools.

2. **Need for Automation**  
    Manual translation is time-consuming and prone to errors, necessitating the development of automated solutions for efficiency.

3. **Quality of Existing Tools**  
    Many existing translation systems fail to capture the nuances of language, leading to inaccurate or contextually incorrect translations.

4. **Scalability Challenges**  
    Traditional translation methods struggle to handle large-scale datasets or real-time translation needs in a cost-effective manner.

5. **Domain-Specific Adaptation**  
    General-purpose translation models may not perform well in specialized domains, highlighting the need for fine-tuned solutions.

### Objective of the Project
1. **Develop an Accurate Translation System**  
    Build a reliable English-to-French translation model that minimizes errors and captures linguistic nuances effectively.

2. **Leverage Pre-trained Models**  
    Utilize the pre-trained MarianMT model to reduce training time and improve translation quality by fine-tuning it on a specific dataset.

3. **Enhance Accessibility**  
    Provide a user-friendly and efficient solution for individuals and organizations requiring English-to-French translations.

4. **Optimize for Performance**  
    Fine-tune the model with appropriate training parameters to achieve high accuracy and scalability for real-world applications.

5. **Facilitate Domain-Specific Translation**  
    Adapt the model to handle specialized domains, ensuring contextually accurate translations for targeted use cases.

### Scope of the Project

1. **Fine-Tuning Pre-trained Models**  
    The project focuses on fine-tuning the MarianMT model to improve translation accuracy for English-to-French tasks.

2. **Custom Dataset Utilization**  
    A custom dataset of English and French sentence pairs is used to train and evaluate the model, ensuring relevance to the target domain.

3. **Scalable Translation Solution**  
    The trained model is designed to handle large-scale datasets and real-time translation needs efficiently.

4. **Domain-Specific Adaptation**  
    The project aims to adapt the translation model for specific domains, enhancing its contextual understanding and accuracy.

5. **Reusable Model and Tokenizer**  
    The fine-tuned model and tokenizer are saved for future use, enabling seamless integration into other applications or workflows.

### Methodology Adopted
1. **Dataset Preprocessing**  
    The English-French dataset is cleaned, formatted, and converted into a Hugging Face Dataset. It is tokenized using the MarianTokenizer and split into training and evaluation subsets.

2. **Model Selection**  
    The pre-trained "Helsinki-NLP/opus-mt-en-fr" MarianMT model is chosen for its proven performance in English-to-French translation tasks.

3. **Fine-Tuning**  
    The MarianMT model is fine-tuned using the Seq2SeqTrainer class with carefully configured training arguments, including learning rate, batch size, and number of epochs.

4. **Evaluation**  
    The model's performance is evaluated on a test dataset using metrics like BLEU scores to ensure translation quality and accuracy.

5. **Model Saving and Deployment**  
    The fine-tuned model and tokenizer are saved locally, enabling easy deployment and reuse for future translation tasks or integration into applications.

### Background Study
### Background Study

1. **Machine Translation Evolution**  
    Machine translation has evolved significantly, from rule-based systems to statistical methods and now neural machine translation (NMT), which leverages deep learning for improved accuracy.

2. **Hugging Face Transformers**  
    The Hugging Face Transformers library provides state-of-the-art pre-trained models, including MarianMT, which simplifies the development of translation systems.

3. **MarianMT Model**  
    Developed by the Helsinki-NLP group, MarianMT models are optimized for multilingual translation tasks, offering robust performance across various language pairs.

4. **Seq2Seq Architecture**  
    Sequence-to-sequence models, such as MarianMT, use encoder-decoder architectures to handle input and output sequences of varying lengths, making them ideal for translation tasks.

5. **Importance of Fine-Tuning**  
    Fine-tuning pre-trained models on domain-specific datasets enhances their contextual understanding, improving translation quality for specialized applications.

### Existing System

1. **Rule-Based Machine Translation (RBMT)**  
    Traditional translation systems that rely on linguistic rules and dictionaries to perform translations. These systems require extensive manual effort to define grammar rules and vocabulary.

2. **Statistical Machine Translation (SMT)**  
    A method that uses statistical models based on bilingual text corpora to predict translations. SMT systems, such as Google Translate in its earlier versions, rely heavily on phrase-based translation techniques.

3. **Example-Based Machine Translation (EBMT)**  
    A system that translates by comparing input sentences with examples from a database of previously translated sentence pairs. It uses analogy-based reasoning to generate translations.

4. **Hybrid Machine Translation**  
    Combines rule-based and statistical approaches to leverage the strengths of both methods, aiming to improve translation accuracy and fluency.

5. **Phrase-Based Translation Systems**  
    A subset of SMT that focuses on translating phrases rather than individual words, improving contextual understanding compared to word-based systems.

### Drawbacks of Existing Systems
### Drawbacks of Existing Systems

1. **Lack of Contextual Understanding**  
    Many traditional systems struggle to capture the context of sentences, leading to translations that are grammatically correct but semantically inaccurate.

2. **Inability to Handle Complex Sentences**  
    Existing systems often fail to translate long or complex sentences effectively, resulting in fragmented or incorrect translations.

3. **Domain-Specific Limitations**  
    General-purpose translation models perform poorly in specialized domains, as they lack the necessary training on domain-specific vocabulary and context.

4. **Resource Intensive**  
    Rule-based and statistical systems require significant computational resources and extensive datasets to achieve acceptable performance levels.

5. **Limited Scalability**  
    Traditional systems are not well-suited for large-scale or real-time translation tasks, making them impractical for modern applications.

6. **Error Propagation**  
    In systems like SMT, errors in earlier stages of translation can propagate, compounding inaccuracies in the final output.

7. **Dependency on Large Parallel Corpora**  
    Statistical and example-based systems rely heavily on large bilingual datasets, which may not be available for all language pairs or domains.

### Proposed System

1. **Neural Machine Translation (NMT)**  
    Leverages the MarianMT model, a state-of-the-art neural machine translation system, to provide accurate and context-aware English-to-French translations.

2. **Fine-Tuning on Custom Dataset**  
    Fine-tunes the pre-trained MarianMT model on a domain-specific English-French dataset to enhance translation quality and contextual understanding.

3. **Efficient Training and Evaluation**  
    Utilizes the Seq2SeqTrainer class with optimized training parameters, ensuring efficient training and evaluation processes for high performance.

4. **Scalable and Reusable Solution**  
    Saves the fine-tuned model and tokenizer for future use, enabling scalability and easy integration into various applications or workflows.

5. **Real-Time Translation Capability**  
    Designed to handle real-time translation tasks efficiently, making it suitable for large-scale and time-sensitive applications.

### Advantages of Proposed System
1. **Improved Translation Accuracy**  
    By fine-tuning the MarianMT model on a custom dataset, the system achieves higher accuracy and better contextual understanding compared to traditional methods.

2. **Domain-Specific Adaptation**  
    The proposed system is tailored to handle specialized domains, ensuring translations are contextually relevant and accurate for specific use cases.

3. **Scalability and Efficiency**  
    The model is designed to handle large-scale datasets and real-time translation tasks, making it suitable for modern applications with high performance demands.

4. **Reusability and Flexibility**  
    The fine-tuned model and tokenizer are saved for future use, allowing seamless integration into other workflows or applications without retraining.

5. **State-of-the-Art Technology**  
    Leveraging the MarianMT model and Hugging Face Transformers library ensures the system benefits from cutting-edge advancements in neural machine translation.

### Steps involved
### Steps Involved

1. **Dataset Preparation**  
    - Load the English-French dataset and clean it by removing null values.
    - Rename columns to "en" (English) and "fr" (French) for consistency.
    - Convert the dataset into a Hugging Face Dataset format.

2. **Tokenization**  
    - Load the MarianTokenizer for the "Helsinki-NLP/opus-mt-en-fr" model.
    - Preprocess the dataset by tokenizing both English and French sentences.
    - Add padding and truncation to ensure uniform input sizes.

3. **Dataset Splitting**  
    - Split the dataset into training and evaluation subsets using an 80-20 or 90-10 ratio.

4. **Model Loading**  
    - Load the pre-trained MarianMT model from Hugging Face.
    - Configure the model for fine-tuning on the English-to-French translation task.

5. **Data Collation**  
    - Use the `DataCollatorForSeq2Seq` to handle dynamic padding during training.

6. **Training Configuration**  
    - Define training arguments such as learning rate, batch size, number of epochs, and evaluation strategy.
    - Enable features like saving checkpoints and generating predictions during evaluation.

7. **Fine-Tuning**  
    - Use the `Seq2SeqTrainer` class to fine-tune the MarianMT model on the prepared dataset.
    - Monitor training progress and evaluate the model's performance after each epoch.

8. **Evaluation**  
    - Evaluate the fine-tuned model on the test dataset using metrics like BLEU scores to measure translation quality.

9. **Model Saving**  
    - Save the fine-tuned model and tokenizer locally for future use.

10. **Deployment**  
     - Deploy the saved model for real-time or batch translation tasks in applications or workflows.
