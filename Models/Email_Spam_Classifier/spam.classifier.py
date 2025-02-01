import numpy as np
import pandas as pd
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Encode labels (Ham = 0, Spam = 1)
encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["label"])

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = " ".join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

# Apply text preprocessing
data["message"] = data["message"].apply(clean_text)

# Tokenization
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(data["message"])
sequences = tokenizer.texts_to_sequences(data["message"])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data["label"], test_size=0.2, random_state=42)

# Build Improved LSTM Model
model = Sequential([
    Embedding(MAX_VOCAB_SIZE, 64, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)),  # Bidirectional LSTM
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Add callbacks (Early Stopping & Model Checkpoint)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("best_spam_model.h5", save_best_only=True, monitor="val_loss")
]

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save final model
model.save("spam_classifier.h5")
print("Spam classifier model saved as 'spam_classifier.h5'")
