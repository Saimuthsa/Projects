# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

def prediction(model , island , bill_length_mm , bill_depth_mm , flipper_length_mm , body_mass_g , sex):
	spec_predicted = model.predict(X)
	spec_predicted = spec_predicted[0]
	if spec_predicted == 0:
		return "Adelie"
	elif spec_predicted == 1:
		return "Chinstrap"
	else:
		return "Gentoo"

st.title("Penguin Species Prediction")
bill_length_mm = st.sidebar.slider("Bill Length = " , float(np.min(df["bill_length_mm"])) , float(np.max(df["bill_length_mm"])))
bill_depth_mm = st.sidebar.slider("Bill Depth = " , float(np.min(df["bill_depth_mm"])) , float(np.max(df["bill_depth_mm"])))
flipper_length_mm = st.sidebar.slider("Flipper Length = " , float(np.min(df["flipper_length_mm"])) , float(np.max(df["flipper_length_mm"])))
body_mass_g = st.sidebar.slider("Body mass = " , float(np.min(df["body_mass_g"]) ), float(np.max(df["body_mass_g"])))
sex = st.sidebar.selectbox("Choose sex :", ("Male" , "Female") )
island = st.sidebar.selectbox("Choose island :" , ("Biscoe", "Dream", "Torgersen"))
type_model = st.sidebar.selectbox("Choose model :" , ("Support Vector Classifier" , "Random Forest Classifier" , "Logistic Regression"))

if st.sidebar.button("Predict"):
	if type_model == "Support Vector Classifier":
		pred = prediction(svc_model , island , bill_length_mm, bill_depth_mm , flipper_length_mm , body_mass_g , sex)
		st.write("Predicted species is :" , pred)
		st.write("Score of the model = " , svc_score)
	elif type_model == "Random Forest Classifier":
		pred_1 = prediction(rf_clf , island , bill_length_mm, bill_depth_mm , flipper_length_mm , body_mass_g , sex)
		st.write("Predicted species is :" , pred_1)
		st.write("Score of the model = " , rf_clf_score)
	else:
		pred_2 = prediction(log_reg , island , bill_length_mm, bill_depth_mm , flipper_length_mm , body_mass_g , sex)
		st.write("Predicted species is :" , pred_2)
		st.write("Score of the model = " , log_reg_score)



