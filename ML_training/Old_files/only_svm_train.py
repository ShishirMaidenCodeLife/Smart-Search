
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Sample data
data = [
    ("text1", "A"),
    ("text2", "A"),
    ("text3", "A"),
    ("text4", "B"),
    ("text5", "B"),
    ("text6", "B"),
    ("text7", "B"),
    ("text8", "C"),
    ("text9", "C"),
    ("text10", "C"),
]

# Separate features (X) and labels (y)
X, y = zip(*data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create an SVM model
model = SVC()

# Train the model
model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
