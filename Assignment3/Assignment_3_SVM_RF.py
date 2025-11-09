import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load data
csv_file = "Dataset.csv"
data = pd.read_csv(csv_file, header = None)

X = data.iloc[:, 0:15].values
y = data.iloc[:, 15].values

# Splitting into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12, stratify = y)

#  Setup for cross-validation
cross_valid = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)
scoring = "accuracy"  

# Random Forest Classifier defined with cross validation
rf = RandomForestClassifier(random_state = 50)

param_grid_rf = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],  # 0.5 = half of features
    "bootstrap": [True]
}

grid_rf = GridSearchCV(
    estimator = rf,
    param_grid = param_grid_rf,
    cv = cross_valid,
    scoring = scoring,
    refit = True,   # refit best model on the full training set
    n_jobs = -1,
    verbose = 0
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test) # prediction for test set using best random forest parameters

print("\n---------------------- RANDOM FOREST  ----------------------")
print("Best params:", grid_rf.best_params_)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# SVM pipeline creation to prevent leakage during dataset standardization
svm_pipe = Pipeline([("scaler", StandardScaler()),("svc", SVC(kernel = "rbf", probability = False, random_state = 42))])

param_grid_svm = {"svc__C": [0.1, 1, 3, 10, 30, 100], "svc__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1.0]}

grid_svm = GridSearchCV(
    estimator = svm_pipe,
    param_grid = param_grid_svm,
    cv = cross_valid,
    scoring = scoring,
    refit = True,
    n_jobs = -1,
    verbose = 0
)

grid_svm.fit(X_train, y_train)

best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print("\n----------------------------- SVM -----------------------------")
print("Best params:", grid_svm.best_params_)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
