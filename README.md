Loan Status Prediction: Machine Learning Classification
This project implements and evaluates six different Machine Learning classification models to predict Loan Status (Approved/Rejected) based on customer profile data.

📊 Dataset Overview
The dataset contains 45,000 records of loan applicants with 14 features, including:

Demographics: Age, Gender, Education, Income.

Financials: Home ownership, Employment experience, Credit score.

Loan Details: Amount, Interest rate, Intent, and Percent of income.

🛠️ Implementation Details
The project is built using Python and the following libraries:

Scikit-Learn: For preprocessing (StandardScaling, OneHotEncoding) and most ML models.

XGBoost: For high-performance gradient boosting.

Pandas/NumPy: For data manipulation.

Models Implemented
Logistic Regression: A baseline linear model for binary classification.

Decision Tree Classifier: A non-linear model that splits data based on feature thresholds.

K-Nearest Neighbors (KNN): A distance-based algorithm (k=5).

Gaussian Naive Bayes: A probabilistic classifier based on Bayes' Theorem.

Random Forest: An ensemble of decision trees using bagging.

XGBoost: An optimized distributed gradient boosting library.


| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC
Logistic Regression | 0.894 | 0.952 | 0.7762 | 0.7418 | 0.7586 | 0.6915
Decision Tree | 0.9017 | 0.8635 | 0.7719 | 0.7945 | 0.7830 | 0.7196
KNN | 0.8938 | 0.9258 | 0.7961 | 0.7050 | 0.7478 | 0.6828
Gauss Naive Bayes | 0.7364 | 0.9362 | 0.4586 | 0.9980 | 0.6284 | 0.5493
Random Forest | 0.9291 | 0.9738 | 0.8933 | 0.7751 | 0.8300 | 0.7887
XGBoost | 0.9358 | 0.9787 | 0.8887 | 0.8144 | 0.8499 | 0.8104

ML Model Name | Observation about model performance

Logistic Regression | Performed well as a baseline with a high AUC (0.952). However, it assumes a linear relationship between features, which may slightly limit its accuracy compared to ensemble methods.

Decision Tree | Achieved high recall (0.7945), meaning it is good at catching actual defaults, but the lower AUC compared to Random Forest suggests it may be prone to overfitting the training data.

KNN | Shows solid performance but is computationally more expensive as the dataset grows. It relies heavily on feature scaling, and its precision (0.7961) indicates it is cautious about predicting positive classes.

Gaussian Naive Bayes | Notable for having the highest Recall (0.9980) but the lowest Precision (0.4586). This model is extremely "liberal," predicting almost every potential default correctly but triggering many false alarms.

Random Forest | One of the top performers. By averaging multiple decision trees, it significantly reduced the variance seen in the single Decision Tree, resulting in a much higher MCC (0.7887) and Precision.

XGBoost | The Best Overall Model. It achieved the highest Accuracy (93.58%), AUC (0.9787), and MCC (0.8104). Its gradient boosting approach effectively minimized errors that other models missed.


🚀 How to Run
Clone the repository or open the folder in VS Code.

Install dependencies:

Bash
pip install -r requirements.txt
Run the Notebook/Script:
Execute the code blocks sequentially to preprocess data, train the models, and generate the evaluation report.