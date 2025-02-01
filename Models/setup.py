from setuptools import setup, find_packages

setup(
    name="neural_network_models",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "nltk",
        "tqdm",
        "gensim",
        "opencv-python",
        "pillow",
        "implicit",
        "surprise",
        "statsmodels",
        "prophet",
        "transformers"
    ],
)
