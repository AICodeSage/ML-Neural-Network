# MNIST CNN Model

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The model is built using TensorFlow and Keras and is trained with data augmentation for improved generalization.

## Installation Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/mnist-cnn.git
   cd mnist-cnn
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage Details
To train the model, run:
```sh
python train.py
```
To evaluate the model:
```sh
python evaluate.py
```
To serve the model as an API:
```sh
uvicorn app:app --reload
```

## GitHub Actions - Automating Model Training
This repository includes a GitHub Actions workflow to automate training on each commit. The workflow file is located in `.github/workflows/train.yml`.

## Deploying as an API
The trained model is deployed using FastAPI. To start the API server:
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```


