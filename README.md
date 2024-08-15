# customer-behaviour-knn

Introduction:

This code is an implementation of a K-Nearest Neighbors (KNN) classifier in Python, used for predicting customer behavior based on their demographic information and purchase history. The goal is to develop a model that can accurately predict whether a customer is likely to make a purchase or not.

Background:

Customer behavior prediction is a crucial task in marketing and sales, as it helps businesses to identify potential customers, tailor their marketing strategies, and improve customer retention. Traditional methods of customer behavior prediction rely on manual analysis of customer data, which can be time-consuming and prone to errors. Machine learning algorithms like KNN provide a more efficient and accurate way to analyze customer data and make predictions.

K-Nearest Neighbors (KNN) Algorithm:

KNN is a popular supervised learning algorithm used for classification and regression tasks. It works by finding the k most similar instances (nearest neighbors) to a new instance, and using their labels to make a prediction. In this code, KNN is used to classify customers as either likely to make a purchase (1) or not likely to make a purchase (0).

Dataset:

The dataset used in this code consists of customer information, including demographic variables like age, salary, and gender, as well as purchase history variables like price. The dataset is split into training and testing sets, where the training set is used to train the KNN model, and the testing set is used to evaluate its performance.

Code Overview:

The code consists of several steps:

Loading the trained KNN model and scaler from files.
Loading new customer data from a CSV file.
One-hot encoding the gender column.
Concatenating the new customer data with the encoded gender column.
Preparing the features for prediction.
Scaling the features using the loaded scaler.
Making predictions using the loaded KNN model.
Adding the predictions to the new customer data.
Saving the predictions to a CSV file.
Making a prediction for a single row of data.
Libraries Used:

The code uses the following Python libraries:

pickle for loading and saving the trained KNN model and scaler.
pandas for data manipulation and analysis.
numpy for numerical computations.
Conclusion:

This code demonstrates the implementation of a KNN classifier in Python for predicting customer behavior. The code is well-structured, easy to follow, and provides a clear example of how to use KNN for classification tasks.
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
knn_classifier_customer_behaviour.py
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

Here is a step-by-step breakdown of the code:

Step 1: Import necessary libraries

import pandas as pd: Import the pandas library and assign it the alias pd.
import numpy as np: Import the NumPy library and assign it the alias np.
Step 2: Load the dataset

df = pd.read_csv("/content/purchase_history.csv"): Read a CSV file named purchase_history.csv from the /content directory into a pandas DataFrame object called df.
Step 3: Explore the dataset

df.head(): Display the first few rows of the DataFrame df.
df.count(): Display the count of non-null values in each column of the DataFrame df.
len(df): Display the total number of rows in the DataFrame df.
Step 4: Encode the Gender column

gender_encoded = pd.get_dummies(df['Gender']): One-hot encode the Gender column of the DataFrame df using pandas' get_dummies function.
gender_encoded = pd.get_dummies(df['Gender'], drop_first=True): One-hot encode the Gender column, but drop the first column to avoid multicollinearity.
Step 5: Concatenate the encoded Gender column with the original DataFrame

df = pd.concat([df, gender_encoded], axis=1): Concatenate the original DataFrame df with the encoded Gender column gender_encoded along the columns (axis=1).
Step 6: Split the data into features (X) and target (y)

x = df[['Male', 'Age', 'Salary', 'Price']].to_numpy(): Select the columns Male, Age, Salary, and Price from the DataFrame df and convert them to a NumPy array x.
y = df['Purchased'].to_numpy(): Select the column Purchased from the DataFrame df and convert it to a NumPy array y.
Step 7: Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42): Split the data into training sets x_train and y_train, and testing sets x_test and y_test, with a test size of 20% and a random state of 42.
Step 8: Scale the data using StandardScaler

scaler = StandardScaler(): Create a StandardScaler object scaler.
x_train = scaler.fit_transform(x_train): Fit the scaler to the training data x_train and transform it.
x_test = scaler.fit_transform(x_test): Transform the testing data x_test using the same scaler.
Step 9: Train a K-Nearest Neighbors (KNN) classifier

k = 5: Set the number of neighbors to 5.
knn = KNeighborsClassifier(n_neighbors=k): Create a KNN classifier object knn with 5 neighbors.
knn.fit(x_train, y_train): Train the KNN classifier on the training data x_train and y_train.
Step 10: Make predictions on the testing data

y_pred = knn.predict(x_test): Use the trained KNN classifier to make predictions on the testing data x_test.
Step 11: Evaluate the model using accuracy score

accuracy = accuracy_score(y_test, y_pred): Calculate the accuracy score of the model by comparing the predicted values y_pred with the actual values y_test.
Step 12: Save the model and scaler using pickle

with open('knn_model.pickle', 'wb') as f: pickle.dump(knn, f): Save the trained KNN model to a file named knn_model.pickle using pickle.
with open('scaler.pickle', 'wb') as f: pickle.dump(scaler, f): Save the scaler object to a file named scaler.pickle using pickle.
Step 13: List the files in the current directory

!ls: Use the ls command to list the files in the current directory.
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
inferencing.py
 _____________________________________________________________________________________________________________________________________________________________________________________________________________________________
Here is a step-by-step breakdown of the code:

Step 1: Load the KNN model and scaler

with open('/content/knn_model.pickle','rb') as f: knn_new = pickle.load(f): This line loads a pickled KNN model from a file named knn_model.pickle in the /content directory. The model is stored in the knn_new variable.
with open('/content/scaler.pickle','rb') as f: scaler_new = pickle.load(f): This line loads a pickled scaler from a file named scaler.pickle in the /content directory. The scaler is stored in the scaler_new variable.
Step 2: Load the new customer data

new_df = pd.read_csv("/content/new_customers.csv"): This line reads a CSV file named new_customers.csv from the /content directory into a pandas DataFrame object called new_df.
Step 3: Explore the new customer data

new_df: This line displays the new customer data in the new_df DataFrame.
len(new_df): This line displays the total number of rows in the new_df DataFrame.
Step 4: Encode the Gender column

gender_encoded_new = pd.get_dummies(new_df['Gender'], drop_first=True): This line one-hot encodes the Gender column of the new_df DataFrame using pandas' get_dummies function. The resulting encoded column is stored in the gender_encoded_new variable.
Step 5: Concatenate the encoded Gender column with the original DataFrame

df_new_2 = pd.concat([new_df,gender_encoded_new],axis=1): This line concatenates the original new_df DataFrame with the encoded Gender column gender_encoded_new along the columns (axis=1). The resulting DataFrame is stored in the df_new_2 variable.
Step 6: Select the features

x_new = df_new_2[['Male','Age','Salary','Price']].to_numpy(): This line selects the columns Male, Age, Salary, and Price from the df_new_2 DataFrame and converts them to a NumPy array x_new.
Step 7: Scale the features

x_new_scale2 = scaler_new.fit_transform(x_new): This line scales the features x_new using the loaded scaler scaler_new. The scaled features are stored in the x_new_scale2 variable.
Step 8: Make predictions

y_new_pred = knn_new.predict(x_new_scale2): This line uses the loaded KNN model knn_new to make predictions on the scaled features x_new_scale2. The predicted values are stored in the y_new_pred variable.
Step 9: Add the predicted values to the original DataFrame

df_new_2['will_purchase'] = y_new_pred: This line adds the predicted values y_new_pred to the original df_new_2 DataFrame as a new column named will_purchase.
Step 10: Save the results

df_new_2.to_csv("model_predictions.csv",index=False): This line saves the resulting DataFrame df_new_2 to a CSV file named model_predictions.csv in the current directory.
Step 11: Make a prediction for a single row

row_values = [1, 32, 40000,5000]: This line defines a single row of values to make a prediction for.
x_new = np.array(row_values).reshape(1,-1): This line converts the row values to a NumPy array and reshapes it to a 2D array with a single row.
x_new_scale2 = scaler_new.fit_transform(x_new): This line scales the single row of values using the loaded scaler scaler_new.
y_new_pred = knn_new.predict(x_new_scale2): This line uses the loaded KNN model knn_new to make a prediction on the scaled single row of values.
prediction = str(y_new_pred[0]): This line converts the predicted value to a string.
print("Purchase ? "+str(prediction)): This line prints the predicted value as a string.
