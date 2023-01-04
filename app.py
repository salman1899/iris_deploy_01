import streamlit as st
import pickle
import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,classification_report,confusion_matrix


model=pickle.load(open("model_hm.pkl","rb"))

def iris_types(num):
    if num == 0:
        return "setosa"
    elif num == 1:
        return "versicolor"
    else:
        return "virginica"

def main():
    st.title("Iris Classification App")
    menu=["Home","Help","About"]
    choice=st.sidebar.selectbox("Menu",menu)    
    st.image("iris.png")
    if choice == "Home":
        
        num1 = st.number_input(
            "sepal_length",
           min_value=0.0,
           max_value=8.0,
           step=1e-1,
           format="%.1f")
        #st.write("p l is ",num1)
       
        num2 = st.number_input(
           "sepal_width",
           min_value=0.0,
           max_value=8.0,
           step=1e-1,
           format="%.1f")
        #st.write("p w is ",num2)
        
        num3 = st.number_input(
            "petal_length",
           min_value=0.0,
           max_value=8.0,
           step=1e-1,
           format="%.1f")
        #st.write("w l is ",num3)
        
        num4 = st.number_input(
            "petal_width",
           min_value=0.0,
           max_value=8.0,
           step=1e-1,
           format="%.1f")
        #st.write("w p is ",num4)
        
        button = st.button('Submit')
        if button == True:
            a=model.predict([[num1,num2,num3,num4]])
            b=np.argmax(a)
            c=iris_types(b)
                
            st.write(c)
        
        
        
        
        
      
          
          
if __name__ == '__main__':
    main()
    