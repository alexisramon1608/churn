# Customer Churn Prediction for Telecom Company

This project demonstrates how to predict customer churn for a telecom company using logistic regression and cross-validation. The model is built using `scikit-learn` for machine learning and `matplotlib` and `seaborn` for visualizations. The primary goal of the model is to predict whether a customer will churn based on historical data and key features such as customer tenure, monthly charges, and contract type.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)

## Project Overview

This project uses a **logistic regression** model to predict customer churn based on various features. Cross-validation is used to evaluate the model's performance. After training the model, key metrics such as accuracy, classification report, and confusion matrix are calculated and visualized. The project also includes visualizations of the model's cross-validation scores and confusion matrix to better understand its performance.

## Technologies Used
- **Python 3.x**
- **Libraries:**
  - `pandas` - For data manipulation and analysis.
  - `numpy` - For numerical operations.
  - `scikit-learn` - For machine learning models and evaluation.
  - `matplotlib` - For data visualization.
  - `seaborn` - For advanced visualization (used for the confusion matrix heatmap).

## Dataset

The dataset used in this project contains customer information for a telecom company and is assumed to have the following columns:
- `Churn`: Target variable (1 if the customer has churned, 0 if not).
- Other features like `tenure`, `monthly_charges`, `total_charges`, `contract_type`, etc.

The dataset should be a CSV file (e.g., `telecom_churn.csv`). You can replace the example dataset with your own if needed.

## Installation

To run this project, you need Python 3.x and the following libraries. You can install the necessary dependencies using `pip`:

```bash
pip install pandas scikit-learn matplotlib seaborn
