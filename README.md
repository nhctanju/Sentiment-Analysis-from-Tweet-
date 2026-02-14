# A Comparative Sentiment Analysis on Tweets Using Simple and Complex Models

## 📘 Overview

This project presents a comparative study of **sentiment and emotion classification** on Twitter data using both **traditional machine learning models** and **deep learning architectures**. The primary objective of the study is to evaluate how different models perform on **noisy, short, and informal social media text**, and to determine whether more complex, context-aware models provide better classification results.

This was group based project as a part Machine Learning (CSE427) course. It includes a **detailed academic report** and presentation video uploaded on YouTube.

---

## 🎯 Objectives

The main goals of this project were:

- To perform sentiment and emotion classification on Twitter data  
- To compare the performance of traditional machine learning models and deep learning models  
- To analyze the impact of contextual and sequential learning in sentiment analysis  
- To identify which models are better suited for noisy, real-world social media text  

---

## 🗂️ Dataset

- Source: [Publicly available **Kaggle Twitter Sentiment Dataset**](https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets)  
- Content: Tweets annotated with  
  - Sentiment polarity  
  - Emotion categories (e.g., joy, anger, fear, trust)  
- Nature of data:
  - Short text length  
  - Informal language  
  - Slang, abbreviations, and hashtags  

---
## 🔄 Project Workflow

- Data Collection
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Word Embeddings (for Deep Learning models)
- Model Training
- Model Evaluation
- Performance Comparison

---

## 🧠 Models Implemented

A variety of models were used to compare performance across different complexity levels.

### 🔹 Traditional Machine Learning Models
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  

### 🔹 Deep Learning Models
- Simple Recurrent Neural Network (RNN)  
- Bidirectional RNN  
- Long Short-Term Memory (LSTM)  

---

## 📊 Results Summary

The experimental results show that **deep learning models outperform traditional machine learning approaches** for sentiment classification on Twitter data.

| Model | Accuracy |
|------|---------|
| LSTM | **0.7501** |
| Simple RNN | 0.7331 |
| Random Forest | 0.7263 |
| Bidirectional RNN | 0.7201 |
| Naive Bayes | 0.6552 |
| KNN | 0.6600 |

### 🔍 Key Observations

- **LSTM achieved the highest accuracy**, indicating strong performance in capturing contextual dependencies.  
- Sequential neural models (RNN, LSTM) performed better than feature-based classifiers.  
- Traditional models struggled with:
  - Informal language  
  - Slang and abbreviations  
  - Emotionally expressive text  

These findings support existing research that **context-aware and embedding-based models** are more effective for social media sentiment analysis.

---

## 📄 Project Report

- A detailed academic report is included in this repository, containing:
- Full methodology
- Model architectures
- Graph and Visuals
- Detailed results and analysis
- References and related work

📌 For complete technical details, please refer to the project report.

---
## 🎥 Presentation Video (YouTube): 

A presentation video explaining the project, methodology, and results. Watch the full explanation here:

🔗 [Project **Presentaion** Link](https://youtu.be/jEckM4sI6Rs?si=WGAzgf7gA3qfgmlU)

---

## 🛠️ Technologies Used

- Python 🐍
- Scikit-learn
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook 📓

---
## 🎓 Key Learning Outcomes

Through this project, we:

- Compared simple and complex models for sentiment analysis
- Observed the impact of sequential learning on text classification
- Understood the challenges of noisy social media data
- Demonstrated the effectiveness of deep learning models for emotion detection
---
## 👥 Project Team

This was a group project consisting of three members:
- Niamul
- Enayet
- Shifat Sharif

