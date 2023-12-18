from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import csv

csv_data_path="Data/search_link_data.csv"

data=[]

with open (csv_data_path,'r') as csvfile:
    csvreader=csv.reader(csvfile)
    for row in csvreader:
        if len(row)==2:
            data.append(tuple(row))


# Separate features (X) and labels (y)
X, y = zip(*data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# SVC and Logistic MOdels
model1 = SVC()
model2=LogisticRegression()

# Training the models
model1.fit(X_train_vectorized, y_train)
model2.fit(X_train_vectorized,y_train)

# Make predictions on the test set
y_pred1 = model1.predict(X_test_vectorized)
y_pred2 = model2.predict(X_test_vectorized)

# Evaluate the model

accuracy1 = accuracy_score(y_test, y_pred1)
report1 = classification_report(y_test, y_pred1)

accuracy2 = accuracy_score(y_test, y_pred2)
report2 = classification_report(y_test, y_pred2)

# Display results
print(f"Accuracy for svm: {accuracy1:.2f}")
print("Classification Report for svm:\n", report1)

print(f"Accuracy for logistic_reg: {accuracy2:.2f}")
print("Classification Report for logistic_reg:\n", report2)

with open('Models/vectorizer.pkl','wb') as file:
    pickle.dump(vectorizer,file)

with open('Models/model1.pkl','wb') as file:
    pickle.dump(model1,file)

with open('Models/model2.pkl','wb') as file:
    pickle.dump(model2,file)

print("training_completed")



