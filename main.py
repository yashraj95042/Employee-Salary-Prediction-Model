import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

string = "My Prediction"

st.set_page_config(page_title=string,page_icon="📒")
st.title("Employee Salary Prediction")

st.write("""
# Salary Prediction Model
Salary vs. *Experience*
""")

#reading the data using panda library
df = pd.read_csv("Salary.csv")
st.write(df)
# the data has been splitted in train and testing set
x = df.iloc[:,[0]].values
y = df.iloc[:,-1].values

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3,random_state=0)
exp = st.sidebar.slider("Experince",1,13,1)
#Define LinearRegression Model :

lr = LinearRegression()
lr.fit(x_train, y_train)

#Test model :

y_pred = lr.predict([[exp]])
st.write(f"If Experience is : ",exp,"years")
st.write(f"Then Predicted Salary : ",float(y_pred))


st.write("""
# Scatter Plot
Salary vs. *Experience*
""")

fig = plt.figure()

plt.scatter(x,y,alpha=0.8,cmap='viridis')

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.colorbar()
st.pyplot(fig)
