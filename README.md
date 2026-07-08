# Fresher Recruitment Prediction and Candidate Evaluation System using Machine Learning

## Overview

This project builds an **Ensemble Machine Learning classification model** that analyzes a fresh graduate candidate's profile — including technical scores, academic GPA, projects, internships, and attendance — to predict whether the candidate will be **Selected** or **Not Selected** during a fresher hiring process.

## Objective

To develop a robust classification model that automates and supports fresher hiring decisions by learning patterns from historical candidate data, helping recruiters make faster and more consistent shortlisting decisions.

## Dataset

- **Name**: `FRESHER HIRING PROJECT.csv`
- **Description**: Contains candidate-level metrics such as:
  - Age
  - CGPA
  - Technical skills: Programming Score, SQL Score, Aptitude Score
  - Communication Score
  - Extracurricular activities
  - Attendance
  - `Selected` (target variable — hiring outcome)

> Place the dataset file in the same directory as the notebook before running it, or update the file path in the data-import cell.

## Project Workflow

1. **Import Libraries** – pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
2. **Import Data** – Load the CSV dataset and clean up unnamed/index columns
3. **Describe Data** – Explore shape, summary statistics, and data types
4. **Data Visualization**
   - Distribution of candidate selection outcomes
   - Comparison of average scores (CGPA, Aptitude, Programming, SQL, Communication) between selected and non-selected candidates
5. **Data Preprocessing**
   - Handle missing values (median for numeric, mode for categorical)
   - Encode categorical features using `LabelEncoder`
6. **Feature & Target Definition** – Split data into features (`X`) and target (`y = Selected`)
7. **Train-Test Split** – 80% training / 20% testing, stratified on the target
8. **Modeling**
   - Balance class distribution using `RandomOverSampler`
   - Scale features using `StandardScaler`
   - Train a `RandomForestClassifier` (ensemble model, 100 estimators)
9. **Model Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
10. **Prediction Examples** – Demonstrates predictions on sample high-performing and low-performing candidate profiles

## Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – preprocessing, modeling, evaluation
  - `imbalanced-learn` – class balancing (RandomOverSampler)

## Model

- **Algorithm**: Random Forest Classifier (Ensemble Learning)
- **Balancing Technique**: Random Over-Sampling
- **Feature Scaling**: StandardScaler

## How to Run

1. Clone or download this repository.
2. Ensure `FRESHER HIRING PROJECT.csv` is in the working directory.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
4. Open and run `PROJECT.ipynb` in Jupyter Notebook / JupyterLab / Google Colab.
5. Execute the cells sequentially from top to bottom.

## Results

The trained ensemble model is evaluated on the test set using accuracy, a confusion matrix, and a full classification report. Sample predictions are also demonstrated on manually crafted high-performer and low-performer candidate profiles to validate model behavior.

## Explanation

This Machine Learning project takes a fresher's credentials — such as CGPA, test scores, and attendance — handles missing details, and matches them against historical data using an ensemble of decision trees. After balancing data inequalities, it functions as an intelligent selection engine that evaluates any candidate's profile to predict whether they will be Selected or Not Selected.

## Author

**Vaishnavi Vangoori**
Email: vangoorivaishnavi24@ifheindia.org
