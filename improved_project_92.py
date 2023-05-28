import streamlit as st
def calculate_emi(p,no,ro):
    emi=p*(ro/100)*((1+ro/100)**no)/(((1+ro/100)**no)-1)
    return round(emi,3)
st.title("Calculating EMI")
principle = st.slider("Principle amount :", 1000, 100000)
tenure = st.slider("Tenure of the loan :", 1,30)
roi = st.slider("Rate of Interest :",1,15)
n = tenure * 12
r = roi / 12
if st.button("Calculate"):
     emi_calc = calculate_emi(principle , n , r)
     st.write("EMI is = " , emi_calc)


def calculate_outstanding_balance(p,n,r,m):
    balance=p*(((1+r/100)**n) - (1+r/100)**m)/(((1+r/100)**n)-1)
    return balance

st.title("Improving EMI Calculation")
principal = st.slider("New Principle amount :", 1000,100000)
ntenure = st.slider("New Tenure of the loan :", 1,30)
nroi = st.slider("New Rate of Interest :", 1,15)
m = st.slider("Period after which the Outstanding Loan Balance is calculated (in months)" , 1 ,12)
new_n = ntenure * 12
r_new = nroi / 12
if st.button("New Calculate"):
     emi_calc = calculate_emi(principle , new_n , r_new)
     st.write("EMI is = " , emi_calc)
elif st.button("Outstanding Loan Balance"):
    olb = calculate_outstanding_balance(principal , new_n ,r_new ,m)
    st.write("Outstanding Loan Balance = " , olb) 
