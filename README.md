# Quora Duplicate Questions Detection

This project focuses on detecting duplicate questions on Quora using machine learning techniques. It includes data analysis, feature engineering, and model development to predict if pairs of questions are duplicates.

## Project Structure

Repository Structure:

DataSet_Analysis.ipynb # Jupyter notebook for dataset analysis
onlyBOW(bag).ipynb # Jupyter notebook for Bag-of-Words (BOW) approach
AdditionalFeatureModification.ipynb # Jupyter notebook for additional feature engineering
app.py # Streamlit web application for model deployment
helper.py # Helper functions for data preprocessing
cv.pkl # Serialized CountVectorizer object
model.pkl # Serialized RandomForestClassifier model


## Overview

### Problem Statement
Quora's platform aims to provide high-quality answers to questions. Detecting duplicate questions can help maintain content quality and improve user experience by reducing redundancy.

### Dataset
To download dataset , use this link [Kaggle](https://www.kaggle.com/c/quora-question-pairs/data#)

The dataset (`train.csv`) consists of pairs of questions labeled as duplicate or not. It includes features like question text, ID, and labels.

### Approach
1. **Data Analysis**: Initial exploration (`DataSet_Analysis.ipynb`) to understand data distribution, duplicate ratios, and basic statistics.
2. **Feature Engineering**: Added features like question length, word counts, and common word ratios (`AdditionalFeatureModification.ipynb`).
3. **Model Development**: Trained a RandomForestClassifier using Bag-of-Words (BOW) features (`onlyBOW(bag).ipynb`).
4. **Web Application**: Developed a Streamlit app (`app.py`) to interactively predict if two questions are duplicates.

### Files

- `DataSet_Analysis.ipynb`: Analyzes the dataset (`train.csv`) to gain insights into question distribution, duplicates, and basic statistics.

- `onlyBOW(bag).ipynb`: Implements Bag-of-Words (BOW) approach for feature extraction and trains a RandomForestClassifier to predict duplicates.

- `AdditionalFeatureModification.ipynb`: Explores additional features like question length, word counts, and common word ratios to improve model performance.

- `app.py`: Streamlit web application for real-time prediction of duplicate questions using the trained model (`model.pkl` and `cv.pkl`).

- `helper.py`: Contains helper functions for preprocessing and creating input features for the model.

- `cv.pkl` and `model.pkl`: Serialized objects of CountVectorizer and RandomForestClassifier respectively, used for model deployment in `app.py`.

## Usage

### Setup

1. Clone the repository:
   bash

       git clone https://github.com/your-username/quora-duplicate-questions.git
       cd quora-duplicate-questions

3. Install dependencies:

         pip install -r requirements.txt

### Running the Streamlit App

1. Navigate to the project directory:

         cd quora-duplicate-questions

2. Run the Streamlit app:

         streamlit run app.py
3. Access the app in your web browser at `http://localhost:8501`.

## Future Improvements

* Implement advanced natural language processing (NLP) techniques like word embeddings (Word2Vec, GloVe) or deep learning models (LSTM, BERT).
* Enhance user interface and add more interactive features to the Streamlit app.
* Optimize model hyperparameters and explore different ensemble techniques for better performance.

