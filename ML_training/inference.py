def SVC_inference(new_test_data):

    import pickle
    from sklearn.feature_extraction.text import CountVectorizer
    # from default_values import MODEL_DIRECTORY

    import os
    ROOT_DIRECTORY=os.path.dirname(os.path.realpath(__file__))

    MODEL_DIRECTORY=f"{ROOT_DIRECTORY}/Models"
    

    #Load the trained models using pickle
    with open(f'{MODEL_DIRECTORY}/model1.pkl','rb') as file:
        model1=pickle.load(file)

    with open(f'{MODEL_DIRECTORY}/model2.pkl','rb') as file:
        model2=pickle.load(file)

    with open(f'{MODEL_DIRECTORY}/vectorizer.pkl','rb') as file:
        vectorizer=pickle.load(file)

    new_test_data = [new_test_data]
    

    # vectorizer = CountVectorizer()
    # Vectorize the new test data using the same vectorizer
    new_test_data_vectorized = vectorizer.transform(new_test_data)

    # Make predictions on the new test data
    predicted_labels1 = model1.predict(new_test_data_vectorized)
    predicted_labels2 = model2.predict(new_test_data_vectorized)

    print(type(predicted_labels1))
    print(predicted_labels1)
    print(predicted_labels2) ## this is list



    # # Display the predicted labels for the new test data
    # def output_category(predicted_labels):
        
    #     for text, label in zip(new_test_data, predicted_labels):
    #         print(f"Text: '{text}' is predicted to belong to the category: {label}")
    #         return label
            

    # print("SVM result:")
    # print_outputs(predicted_labels1)
    # print("\n Logistic Reg result")
    # print_outputs(predicted_labels2)

    return(predicted_labels1)
    # return(predicted_labels2)


# #To test from here uncomment below.... if using in fastAPI then comment the below code....
# user_srch=input("Enter the test search ")
# SVC_inference(user_srch)




