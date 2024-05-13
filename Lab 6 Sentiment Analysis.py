import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Load your CSV dataset
df = pd.read_csv(r"D:\Dataset\Dataset_sentiment.csv")

# Extract text and sentiment columns
texts = df["text"].values
sentiments = df["sentiment"].values

# Convert sentiment labels to binary (0 for negative, 1 for positive)
sentiments_binary = np.where(sentiments == "positive", 1, 0)

# Tokenize the text data
num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max(map(len, text_sequences))
text_sequences_padded = pad_sequences(text_sequences, maxlen=max_sequence_length)

# Create the CNN model
embedding_dim = 128
kernel_size = 5
filters = 64

model_cnn = Sequential([
    Embedding(input_dim=num_words, output_dim=embedding_dim),
    Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=64, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
batch_size = 64
epochs = 10
model_cnn.fit(text_sequences_padded, sentiments_binary, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Function to preprocess and predict sentiment using CNN model
def predict_sentiment_cnn(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=max_sequence_length)
    sentiment_score = model_cnn.predict(text_sequence)[0][0]
    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"
    
    # Get the embedding layer weights
    embedding_layer = model_cnn.layers[0]
    weights = embedding_layer.get_weights()[0]
    
    # Get the indices of words in the input text
    word_indices = text_sequence[0]
    
    # Get the words corresponding to the indices
    words = [word for word, index in tokenizer.word_index.items() if index in word_indices]
    
    # Check if there are any negative words in the input text
    negative_words = [words[i] for i, index in enumerate(word_indices) if index < len(weights) and weights[index][0] < 0]
    
    if negative_words:
        sentiment = "Negative"
    
    return sentiment, sentiment_score, words


# Input a text for sentiment prediction using CNN model
input_text = input("Enter a text for sentiment prediction: ")
sentiment, sentiment_score, words = predict_sentiment_cnn(input_text)
print(f"Input Text: {input_text}")
print(f"Predicted Sentiment: {sentiment} (Score: {sentiment_score:.4f})")
print(f"Influential Words: {words}")
