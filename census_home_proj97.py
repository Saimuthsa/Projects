import streamlit as st 

def app(df):
	st.header("Census Visulation Prediction ")
	st.text("This web app allows a user to explore and visualise census data")

	st.header("View Data")
	with st.beta_expander("View Full Dataset"):
		st.table(df)

	if st.checkbox("Show summary"):
		st.table(df.describe())


