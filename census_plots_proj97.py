import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
def app(df):
	st.header("Visualise Data")
	st.set_option("deprecation.showPyplotGlobalUse" , False)

	user_choice = st.multiselect("Select Type Of Plot:" , ("Countplot" , "Piechart" , "Boxplot"))

	if "Countplot" in user_choice:
		count_features = st.multiselect("Choose Features for Countplot:" , ('age', 'workclass', 'fnlwgt', 'education', 'education-years', 'marital-status', 'occupation', 'relationship', 'race','gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'))

		for i in range(len(count_features)):
			st.subheader(f"Count plot for {count_features[i]} and income")

			plt.figure(figsize = (20,5))
			sns.countplot(x = df[count_features[i]] , hue = df["income"])
			st.pyplot()
	if "Piechart" in user_choice:
	    pie_features = st.multiselect("Choose Features for Piechart:" , ('age', 'workclass', 'fnlwgt', 'education', 'education-years', 'marital-status', 'occupation', 'relationship', 'race','gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'))
	    for i in range(len(pie_features)):
	    	st.subheader(f"Piechart for {pie_features[i]} ")
	    	plt.figure(figsize = (20,5))
	    	plt.pie(df[pie_features[i]].value_counts(), labels = df[pie_features[i]].unique() , autopct='%1.1f%%')
	    	st.pyplot()
	if "Boxplot" in user_choice:
		box_features = st.multiselect("Choose Features for Boxplot:" , ('age', 'workclass', 'fnlwgt', 'education', 'education-years', 'marital-status', 'occupation', 'relationship', 'race','gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'))

		for i in range(len(box_features)):
			st.subheader(f"Boxplot for {box_features[i]} ")

			plt.figure(figsize = (20,5))
			sns.boxplot(y = df[box_features[i]] , x = df["hours-per-week"])
			st.pyplot()
	
