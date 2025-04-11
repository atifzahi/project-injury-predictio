# project-injury-prediction
Project Overview

This project explores how machine learning models can be used to predict injury risks in athletes based on biometric data, training intensity, and recovery patterns.

Research Question

How can machine learning models predict sports injury risks based on athlete biometrics, training intensity, and recovery patterns?

Dataset

Source: https://www.kaggle.com/datasets/shashwatwork/injury-prediction-for-competitive-runners

Format: CSV
Size: Tabular time-series data of runners with injury status labels

Features Overview
total km, km Z3-4, km Z5-T1-1: total and zonal distance covered

nr. session: number of sessions per day

sprinttr, strength tr.: sprint and strength training

 Recovery Indicators
hours alter, perceived exertion, perceived soreness, perceived fatigue (often repeated per athlete/day)

These are subjective indicators of how tired/recovered the athlete feels


Multiple days are represented as rolling/lag features:
e.g., total km, km Z3-4, sprinttr, etc., repeated as total km.2, total km.3, ..., up to total km.6

Athlete ID: unique ID for each athlete (masked)

Date: time-series tracking

injury: target label (1 = injury, 0 = no injury)
Project Structure & Workflow
1. Data Preprocessing & Visualization
Date conversion, missing value handling

Injury class distribution plot

Top 10 correlated features with injury

Injury counts over time (line plot)

2. Model Training & Evaluation
Models Used:

XGBoost

LightGBM

Extra Trees

Gradient Boosting

Logistic Regression

Key Steps:

Train/test split (80/20)

Feature scaling with StandardScaler

Balanced undersampling (no SMOTE used)

Hyperparameter tuning using GridSearchCV (for Logistic Regression & Extra Trees)

Threshold optimization for each model to boost F1-Score

Evaluation via Accuracy, Precision, Recall, F1, ROC AUC

3. Cross-validation
Stratified 5-fold CV for stability

F1 CV average included in results

4. Visuals
ROC Curve for each model

Final bar plot comparing F1 Scores

Future Improvements
Try deep learning models (e.g., RNN/LSTM for time-series)

Integrate wearable device real-time monitoring

Develop a visual injury-risk dashboard
License
This project is for academic use only. All rights reserved.

















