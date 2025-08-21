# NBA Player Shot Prediction

This project builds a machine learning pipeline to predict whether a given NBA player will make a shot based on game context and shot details. Using play-by-play and shot chart data, the model incorporates features such as:

- Time left (game clock, shot clock)

- Action type (jump shot, layup, dunk, etc.)

- Shot zone area (left wing, right corner, paint, etc.)

- Shot distance and angle

The notebook demonstrates an end-to-end workflow:

# Features

- Data Collection: Retrieves play-by-play and shot data via the NBA API

- Data Preprocessing:
  - Encodes categorical features (action type, shot zone, etc.)

  - Normalizes numeric features (distance, angle, time left) using Scikit-learn

  - Handles missing values and categorical transformations

- Modeling:

  - Trains an XGBoost Classifier to predict shot success (make vs. miss)

  - Baseline comparisons with logistic regression or random forest (optional)

- Evaluation:

  - Accuracy, confusion matrix, precision, recall, ROC-AUC

  - Feature importance to interpret shot-making patterns

- Visualization:

  - Shot probability heatmaps by zone

  - Distribution plots of shot attempts and makes

  - Context-conditioned probability plots (e.g., by time remaining or defender proximity)

# Tech Stack

Python

Pandas, NumPy — data handling and feature engineering

Scikit-learn — preprocessing, metrics, baseline models

XGBoost — gradient-boosted trees classifier

Matplotlib — exploratory visualization and probability heatmaps

nba_api — data collection
