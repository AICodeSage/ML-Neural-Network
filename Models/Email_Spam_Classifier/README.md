# Spam Classifier using LSTM

## Overview
This project is a spam classifier built using a Bidirectional LSTM (Long Short-Term Memory) neural network. The model processes text messages to determine whether they are spam or not, using the **SMS Spam Collection** dataset.

## Dataset
The dataset used is **spam.csv**, which contains SMS messages labeled as either spam or ham (not spam). The dataset is preprocessed and tokenized before being passed into the LSTM model.

## Features
- **Text Preprocessing:**
  - Converts text to lowercase
  - Removes special characters and extra spaces
  - Removes stopwords
- **Tokenization & Padding:**
  - Tokenizes text using Keras Tokenizer
  - Pads sequences to a fixed length
- **Deep Learning Model:**
  - Embedding Layer for word representation
  - Bidirectional LSTM with dropout
  - Fully connected layers for classification
- **Training & Evaluation:**
  - Uses early stopping and model checkpointing
  - Achieves high accuracy in detecting spam messages

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas tensorflow nltk scikit-learn
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier
   ```
2. Download the dataset (`spam.csv`) and place it in the project folder.
3. Run the script:
   ```bash
   python spam_classifier.py
   ```

## Model Training & Evaluation
The model is trained on the dataset and achieves a high accuracy in classifying spam messages. Training includes:
- **Early stopping** to prevent overfitting
- **Model checkpointing** to save the best-performing model

After training, the model is saved as `spam_classifier.h5`.

## Results
The model's performance is evaluated on a test set, displaying the test accuracy:
```bash
Test Accuracy: 0.XXXX  # (Example output)
```

## Future Improvements
- Implement more advanced NLP techniques (e.g., Transformer-based models)
- Experiment with different embeddings (e.g., Word2Vec, GloVe, BERT)
- Deploy the model as a web app or API

## License
This project is open-source and available under the MIT License.

## Author
Your Name - [AICodeSage](https://github.com/AICodeSage/)

