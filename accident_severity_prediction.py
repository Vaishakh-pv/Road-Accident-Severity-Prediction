''' Road Accident Severity Prediction '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data set
df = pd.read_csv(r"C:\Users\netha\Downloads\archive (4)\accident_prediction_india.csv")

#Uderstanding data set
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# Data Cleaning
df.drop_duplicates(inplace=True)

# Handling missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nAfter Cleaning:")
print(df.isnull().sum())

# Target Variable Identification
for col in df.columns:
    if 'severity' in col.lower():
        target = col
        break
print("\nTarget Column:", target)

# Exploratory Data Analysis
plt.figure(figsize=(7,5))
sns.countplot(
    data=df,
    x=target,
    hue=target,
    palette="Set2",
    legend=False,
    edgecolor="black"
)
plt.title("Distribution of Accident Severity", fontsize=14, fontweight="bold")
plt.xlabel("Severity Level")
plt.ylabel("Number of Accidents")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()



from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Encode Categorical Variables
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])

# Correlation Heatmap
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(
    corr,
    cmap="RdBu_r",
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.show()



# Converting Accident Severity into Binary Classes

df[target] = df[target].apply(lambda x: 1 if x >= 1 else 0)

# Feature and Target Split
X = df.drop(columns=[target])
y = df[target]
X = X.drop(columns=['Number of Fatalities', 'Number of Casualties'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Feature Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred) * 100)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance.head(10),
    x='Importance',
    y='Feature',
    hue='Feature',      
    palette='viridis',
    legend=False,
    edgecolor='black'
)

plt.title("Top 10 Important Features (Random Forest)", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix", fontsize=14, fontweight="bold")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred) * 100)
print(confusion_matrix(y_test, lr_pred))

# K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("\nKNN Accuracy:",
      accuracy_score(y_test, knn_pred) * 100)
print(confusion_matrix(y_test, knn_pred))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, dt_pred) * 100)
print(confusion_matrix(y_test, dt_pred))

dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(
    data=dt_importance.head(10),
    x='Importance',
    y='Feature',
    hue='Feature',
    palette='magma',
    legend=False,
    edgecolor='black'
)
plt.title("Top 10 Features – Decision Tree", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Support Vector Machine
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\nSVM Accuracy:",
      accuracy_score(y_test, svm_pred) * 100)
print(confusion_matrix(y_test, svm_pred))

# Model Comparison
models = [ 'Random Forest', 'Logistic Regression', 'KNN', 'Decision Tree', 'SVM']
accuracy = [
    accuracy_score(y_test, rf_pred ),
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, knn_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, svm_pred)
]

plt.figure(figsize=(9,5))

sns.barplot(
    x=models,
    y=accuracy,
    hue=models,          
    palette="coolwarm",
    legend=False,
    edgecolor="black"
)

plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=20)

# Show values on bars
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)

plt.show()



# Best Model Selection
best_model = models[np.argmax(accuracy)]
print("\nBest Performing Model:", best_model)
