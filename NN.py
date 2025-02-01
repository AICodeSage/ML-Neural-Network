import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Configuration settings
BATCH_SIZE = 128
EPOCHS = 10
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
MODEL_SAVE_PATH = "mnist_cnn_model.h5"

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, *IMG_SHAPE).astype("float32") / 255.0
x_test = x_test.reshape(-1, *IMG_SHAPE).astype("float32") / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Data augmentation
data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Define the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=IMG_SHAPE),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer="adam", 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

# Build and train the model
model = build_model()
history = model.fit(
    data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE), 
    validation_data=(x_test, y_test), 
    epochs=EPOCHS, 
    verbose=2
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.show()

plot_history(history)

# Save the model
if not os.path.exists("models"):
    os.makedirs("models")
model.save(os.path.join("models", MODEL_SAVE_PATH))
print(f"Model saved as '{MODEL_SAVE_PATH}'")
