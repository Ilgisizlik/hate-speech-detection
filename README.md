# Hate Speech Detection using Classical NLP Models

This repository contains the code and experiments for an NLP project on hate speech detection. 
The goal is to compare baseline machine learning models with hyperparameter-optimized versions
across multiple benchmark datasets.

## Datasets
The following datasets were used in this project:

- **Davidson et al. (2017)** Hate Speech Dataset  
  https://github.com/t-davidson/hate-speech-and-offensive-language

- **HASOC 2019 English Dataset**  
  https://hasocfire.github.io/hasoc/2019/

- **Dynamically Generated Hate Dataset (v0.2.3)**  
  https://huggingface.co/datasets/dynamically_generated_hate_speech

Due to licensing and size constraints, datasets are not included in this repository.

## Methodology
1. Text cleaning and normalization
2. Feature extraction using TF-IDF
3. Baseline models:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear SVM
4. Hyperparameter Optimization using GridSearchCV
5. Evaluation using Accuracy and Macro F1-score

## Repository Structure
- `notebooks/`: Jupyter notebooks for each dataset
- `data/`: Dataset references and descriptions
- `figures/`: Result visualizations
- `requirements.txt`: Python dependencies

## Reproducibility
To reproduce the experiments:

```bash
pip install -r requirements.txt
