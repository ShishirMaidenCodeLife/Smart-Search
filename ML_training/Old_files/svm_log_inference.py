import pickle
from sklearn.feature_extraction.text import CountVectorizer


#Load the trained models using pickle
with open('Models/model1.pkl','rb') as file:
    model1=pickle.load(file)

with open('Models/model2.pkl','rb') as file:
    model2=pickle.load(file)

with open('Models/vectorizer.pkl','rb') as file:
    vectorizer=pickle.load(file)

new_test_data = [
    "create a new inventory",
    "modify existing organization details",
    "delete a specific item from inventory",
    "add a user to a specific role",
    "configure database settings",
    "backup data for the current month",
    "import records from a CSV file",
    "export data to a spreadsheet",
    "analyze recent data trends",
    "generate a performance report",
    "create a new user profile",
    "manage user accounts and permissions"
]

# vectorizer = CountVectorizer()
# Vectorize the new test data using the same vectorizer
new_test_data_vectorized = vectorizer.transform(new_test_data)

# Make predictions on the new test data
predicted_labels1 = model1.predict(new_test_data_vectorized)
predicted_labels2 = model2.predict(new_test_data_vectorized)

# Display the predicted labels for the new test data
def print_outputs(predicted_labels):
    for text, label in zip(new_test_data, predicted_labels):
        print(f"Text: '{text}' is predicted to belong to the category: {label}")
        

print("SVM result:")
print_outputs(predicted_labels1)
print("\n Logistic Reg result")
print_outputs(predicted_labels2)




