
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Remove duplicates
df = df.drop_duplicates().copy()

# Encode labels
df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_message"] = df["message"].apply(clean_text)

# Input and target
X = df["clean_message"]
y = df["label_num"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Vectorization Completed!")
print("Training Shape:", X_train_tfidf.shape)
print("Testing Shape:", X_test_tfidf.shape)

# Machine Learning Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []
predictions = {}

# Train and evaluate
for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    ))

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    "output/model_comparison.csv",
    index=False
)

print("\nModel Comparison:")
print(results_df)

# Confusion Matrix for both models
for name, y_pred in predictions.items():

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"output/{filename}_confusion_matrix.png"
    )
    plt.show()

# Save test predictions
prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Naive_Bayes": predictions["Naive Bayes"],
    "Logistic_Regression": predictions["Logistic Regression"]
})

prediction_df.to_csv(
    "output/test_predictions.csv",
    index=False
)

print("\nModel Training and Evaluation Completed!")


# Select best model
best_model = models["Naive Bayes"]

# Function to predict new messages
def predict_message(message):

    cleaned = clean_text(message)

    transformed = tfidf.transform([cleaned])

    prediction = best_model.predict(transformed)[0]

    probability = best_model.predict_proba(transformed)[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM (NOT SPAM)"

    print("\nMessage:", message)
    print("Prediction:", result)
    print("Spam Probability:", round(probability[1] * 100, 2), "%")

# Test messages
predict_message("Congratulations! You have won a free prize. Call now!")
predict_message("Hey, are we meeting tomorrow?")

# Select Best Model
best_model = models["Naive Bayes"]

# Predict New Messages
def predict_message(message):
    cleaned = clean_text(message)
    transformed = tfidf.transform([cleaned])

    prediction = best_model.predict(transformed)[0]
    probability = best_model.predict_proba(transformed)[0]

    result = "SPAM" if prediction == 1 else "HAM (NOT SPAM)"

    print("\nMessage:", message)
    print("Prediction:", result)
    print("Spam Probability:", round(probability[1] * 100, 2), "%")


# Test Messages
predict_message("Congratulations! You have won a free prize. Call now!")
predict_message("Hey, are we meeting tomorrow?")
