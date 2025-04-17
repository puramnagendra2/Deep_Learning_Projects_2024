```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

#read dataset
with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

import re

# Preprocess the text
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
cleaned_text = preprocess_text(text)
print(cleaned_text)

#Tokenizer process
tokenizer = Tokenizer()
#fit
tokenizer.fit_on_texts([text])
#assign length of word index
total_words = len(tokenizer.word_index) + 1

#declare ngrams
input_sequences = []
#split the sentence from '\n'
for line in text.split('\n'):
    #get tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

#maximum sentence length
max_sequence_len = max([len(seq) for seq in input_sequences])
# input sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

#convert one-hot-encode
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

from tensorflow.keras.layers import Bidirectional, Dropout

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)

model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stop, reduce_lr], verbose=1)

#determine a text
seed_text = "I will close the door if"
# predict word number
next_words = 7

for _ in range(next_words):
    #convert to token
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    #path sequences
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #model prediction
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    # get predict words
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

# Predict the next word
token_list = pad_sequences([token_list[0]], maxlen=max_sequence_len-1, padding='pre')
predicted = np.argmax(model.predict(token_list), axis=-1)

# Find the word corresponding to the predicted token
next_word = ""
for word, index in tokenizer.word_index.items():
    if index == predicted:
        next_word = word
        break

print(next_word)

model.save('text_generation_model.keras')

import pickle

# Save the tokenizer to a .pkl file
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
```

### Overview of the Project

1. **Objective**: This project focuses on building a deep learning model for next-word prediction using a dataset of text. The model predicts the next word in a sequence based on the input text.

2. **Data Preprocessing**: The text data is cleaned by removing special characters, numbers, and extra spaces. It is then tokenized and converted into sequences for training.

3. **Model Architecture**: A sequential model is implemented using an embedding layer, bidirectional LSTM layers, dropout layers for regularization, and a dense layer with a softmax activation function for word prediction.

4. **Training Process**: The model is trained using categorical cross-entropy loss and the Adam optimizer. Early stopping and learning rate reduction callbacks are used to optimize training.

5. **Output and Saving**: The trained model predicts the next word(s) based on a seed text. The model and tokenizer are saved for future use in text generation tasks.

### Problem Statement

1. **Text Prediction Challenge**: Predicting the next word in a sequence is a complex task that requires understanding the context and structure of the input text.

2. **Data Complexity**: The dataset may contain noise, such as special characters, numbers, and inconsistent formatting, which needs to be addressed during preprocessing.

3. **Sequence Modeling**: Building a model capable of handling variable-length input sequences and capturing long-term dependencies in text.

4. **Overfitting Risk**: Ensuring the model generalizes well to unseen data by applying regularization techniques and monitoring training performance.

5. **Practical Application**: Developing a reusable and efficient model for real-world text generation tasks, such as autocomplete or chatbot systems.

### Objectives of the Project

1. **Next-Word Prediction**: Develop a deep learning model capable of predicting the next word in a sequence based on the given input text.

2. **Context Understanding**: Train the model to understand the context and semantics of the input text for accurate predictions.

3. **Efficient Preprocessing**: Implement robust text preprocessing techniques to clean and prepare the dataset for training.

4. **Model Optimization**: Design and train a model architecture that balances accuracy and computational efficiency.

5. **Real-World Applications**: Create a reusable model for practical applications like text autocompletion, chatbots, and content generation.

### Scope of the Project

1. **Text Autocompletion**: The model can be integrated into applications like search engines, email clients, or IDEs to provide intelligent text autocompletion.

2. **Chatbot Development**: Enhance chatbot systems by enabling them to predict and generate contextually relevant responses.

3. **Content Generation**: Assist in generating creative content, such as writing prompts, story continuation, or article drafting.

4. **Language Learning Tools**: Support language learners by predicting and suggesting the next word in sentences, improving vocabulary and grammar skills.

5. **Customizable Applications**: Adapt the model for domain-specific text prediction tasks, such as legal document drafting or medical report generation.

### Key Points About the Algorithm Used

1. **Sequence Modeling with LSTMs**: The algorithm leverages Long Short-Term Memory (LSTM) networks, which are well-suited for capturing long-term dependencies in sequential data, making them ideal for text prediction tasks.

2. **Bidirectional LSTMs**: By using bidirectional LSTM layers, the model processes the input sequence in both forward and backward directions, enhancing its ability to understand context.

3. **Embedding Layer**: The embedding layer converts words into dense vector representations, capturing semantic relationships between words and reducing the dimensionality of the input.

4. **Regularization Techniques**: Dropout layers are incorporated to prevent overfitting by randomly deactivating neurons during training, ensuring better generalization.

5. **Categorical Cross-Entropy Loss**: The model is trained using categorical cross-entropy loss, which is effective for multi-class classification tasks like predicting the next word from a vocabulary.

### Background Study

1. **Natural Language Processing (NLP)**: The project builds upon foundational concepts in NLP, including tokenization, text preprocessing, and sequence modeling, which are essential for understanding and generating human language.

2. **Recurrent Neural Networks (RNNs)**: LSTMs, a type of RNN, are used in this project due to their ability to capture long-term dependencies in sequential data, addressing the vanishing gradient problem common in traditional RNNs.

3. **Word Embeddings**: Techniques like word2vec and GloVe have demonstrated the importance of representing words as dense vectors, enabling models to capture semantic relationships. This project uses an embedding layer to learn such representations.

4. **Sequence-to-Sequence Models**: Inspired by advancements in machine translation and text generation, this project applies sequence-to-sequence modeling principles to predict the next word in a given context.

5. **Regularization in Deep Learning**: The use of dropout layers and callbacks like early stopping reflects best practices in preventing overfitting, ensuring the model generalizes well to unseen data.

### Comparison with Existing Systems

1. **N-gram Models**  
    **Introduction**: Predicts the next word based on fixed-length word sequences.  
    **Limitations**: Struggles with long-term dependencies and context understanding; limited by fixed n-gram size.

2. **Markov Chains**  
    **Introduction**: Uses probabilistic transitions between states for text prediction.  
    **Limitations**: Ignores semantic meaning and long-range dependencies; relies heavily on training data patterns.

3. **Rule-Based Systems**  
    **Introduction**: Employs predefined linguistic rules for text generation.  
    **Limitations**: Lacks adaptability to diverse contexts; requires extensive manual rule creation and maintenance.

4. **Basic RNNs**  
    **Introduction**: Sequential models for text prediction without advanced memory mechanisms.  
    **Limitations**: Suffers from vanishing gradient issues, limiting its ability to capture long-term dependencies.

### Proposed System

1. **Advanced LSTM Architecture**: The proposed system utilizes a bidirectional LSTM architecture with dropout layers to effectively capture long-term dependencies and reduce overfitting, ensuring robust text predictions.

2. **Dynamic Sequence Handling**: By leveraging padding and tokenization, the system can handle variable-length input sequences, making it adaptable to diverse text inputs.

3. **Context-Aware Predictions**: The embedding layer enables the model to learn semantic relationships between words, allowing for contextually accurate next-word predictions.

4. **Optimized Training Process**: Incorporates early stopping and learning rate reduction callbacks to enhance training efficiency and prevent overfitting.

5. **Reusable and Scalable Design**: The trained model and tokenizer are saved for future use, enabling seamless integration into various applications like chatbots, text autocompletion, and content generation.

### Advantages of the Proposed System

1. **Enhanced Context Understanding**: The use of bidirectional LSTMs and embedding layers allows the model to capture both forward and backward context, resulting in more accurate and meaningful predictions.

2. **Adaptability to Variable-Length Inputs**: By employing padding and tokenization, the system can handle input sequences of varying lengths, making it versatile for different text prediction tasks.

3. **Reduced Overfitting**: Regularization techniques like dropout layers and callbacks such as early stopping ensure the model generalizes well to unseen data, minimizing overfitting risks.

4. **Scalability and Reusability**: The trained model and tokenizer can be saved and reused across multiple applications, reducing the need for retraining and enabling easy integration into diverse systems.

5. **Improved Training Efficiency**: The inclusion of learning rate reduction and early stopping optimizes the training process, reducing computational costs while maintaining high accuracy.

### Steps Involved in Developing the Model

1. **Dataset Preparation**:
    - Collect and organize the text dataset.
    - Load the dataset into the program using appropriate file handling techniques.

2. **Text Preprocessing**:
    - Clean the text by removing special characters, numbers, and extra spaces.
    - Convert the text to lowercase for uniformity.
    - Tokenize the text into sequences of words.

3. **Tokenization and Sequence Generation**:
    - Use a tokenizer to create a word index and convert the text into numerical sequences.
    - Generate n-gram sequences from the tokenized text for training.

4. **Padding and Input Preparation**:
    - Pad the sequences to ensure uniform length.
    - Split the sequences into input (X) and output (y) for training.
    - One-hot encode the output labels for multi-class classification.

5. **Model Design**:
    - Define a sequential model with the following layers:
      - Embedding layer to learn word representations.
      - Bidirectional LSTM layers for capturing context in both directions.
      - Dropout layers for regularization.
      - Dense layer with softmax activation for word prediction.

6. **Model Compilation**:
    - Compile the model using categorical cross-entropy as the loss function and Adam as the optimizer.
    - Specify accuracy as the evaluation metric.

7. **Training the Model**:
    - Train the model using the prepared input and output data.
    - Use callbacks like early stopping and learning rate reduction to optimize training.

8. **Prediction and Testing**:
    - Test the model by providing a seed text and generating the next word(s).
    - Evaluate the model's performance on unseen data.

9. **Model Saving**:
    - Save the trained model in a suitable format (e.g., `.keras`).
    - Save the tokenizer for future use in text preprocessing.

