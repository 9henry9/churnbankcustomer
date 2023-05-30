import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the customer churn dataset
df = pd.read_csv("C:/Users/DELL/Desktop/churn.csv")

# Remove unnecessary columns and handle missing values
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df = df.dropna()

# Convert non-numeric values to numeric values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.Categorical(df[col]).codes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

# Normalize the input features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Select the most important features using SelectKBest and chi-squared test
selector = SelectKBest(chi2, k=5)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Define the deep learning model architecture
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=50)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model using accuracy, precision, recall, and F1-score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


#plots and charts
churn_by_geo = df.groupby('Geography')['Exited'].sum()
plt.bar(churn_by_geo.index, churn_by_geo.values)
plt.title("Churned Customers by Geography")
plt.xlabel("Geography")
plt.ylabel("Count")
plt.show()



plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="YlGnBu")
plt.title("Correlation Heatmap")
plt.show()

plt.hist(df['Age'], bins=20)
plt.title("Histogram of Customer Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Plot the accuracy
plt.bar(["Accuracy", "Precision", "Recall", "F1-score"], [accuracy, precision, recall, f1])
plt.title("Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.show()

# Print the evaluation metrics
print("Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}".format(accuracy, precision, recall, f1))