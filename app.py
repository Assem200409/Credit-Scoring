import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from assem.class_preparing_5_ import Preparing
#from sklearn.neighbors import KNeighborsClassifier # Импорт модели

st.title("Credit Scoring in Taiwan 2005")


my_data = st.sidebar.selectbox("Choose your dataset",
                 ["UCI_Credit_Card.csv"]
                  ) 

if st.button("Load", disabled = False ):
    df = pd.read_csv( my_data )
    prep = Preparing(df)                # в переменной prep создаем экземпляр класса Preparing
    prep.warning()                      # переименовываем колонку на "default"
    prep.rename_columns()               # выгружаем переименованный датасет
    prep.select_columns()
    prep.group_clients()
    X_all_new, y_new = prep.X_all_new, prep.y_new
    st.write('ok')


limit_ball = st.slider('LIMIT_BALL', min_value=0, max_value=1000000, step=100, format=None)        
pd1 = st.slider('Amount of days(default>0, nodefault<=0) 1-th month', min_value=-10, max_value=10, step=1, format=None)
pd2 = st.slider('Amount of days(default>0, nodefault<=0) 2-th month', min_value=-10, max_value=10, step=1, format=None)
pd3 = st.slider('Amount of days(default>0, nodefault<=0) 3-th month', min_value=-10, max_value=10, step=1, format=None)
pd4 = st.slider('Amount of days(default>0, nodefault<=0) 4-th month', min_value=-10, max_value=10, step=1, format=None)
pd5 = st.slider('Amount of days(default>0, nodefault<=0) 5-th month', min_value=-10, max_value=10, step=1, format=None)
pd6 = st.slider('Amount of days(default>0, nodefault<=0) 6-th month', min_value=-10, max_value=10, step=1, format=None)

b1 = st.slider('bill 1-th month', min_value=0, max_value=100000, step=100, format=None)
b2 = st.slider('bill 2-th month', min_value=0, max_value=100000, step=100, format=None)
b3 = st.slider('bill 3-th month', min_value=0, max_value=100000, step=100, format=None)
b4 = st.slider('bill 4-th month', min_value=0, max_value=100000, step=100, format=None)
b5 = st.slider('bill 5-th month', min_value=0, max_value=100000, step=100, format=None)
b6 = st.slider('bill 6-th month', min_value=0, max_value=100000, step=100, format=None)

p1 = st.slider('payment 1-th month', min_value=0, max_value=100000, step=100, format=None)
p2 = st.slider('payment 2-th month', min_value=0, max_value=100000, step=100, format=None)
p3 = st.slider('payment 3-th month', min_value=0, max_value=100000, step=100, format=None)
p4 = st.slider('payment 4-th month', min_value=0, max_value=100000, step=100, format=None)
p5 = st.slider('payment 5-th month', min_value=0, max_value=100000, step=100, format=None)
p6 = st.slider('payment 6-th month', min_value=0, max_value=100000, step=100, format=None)


st.success( f"credit = {limit_ball}" )
st.title("Predictions")
col1, col2, col3, col4 = st.columns(4)

if col1.button("KNeighbors", disabled = False ):    
    with open("./knn.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        knn = load( f )                # загруженная модель knn
        exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

        pred = knn.predict([exp])[0]
        pred_proba = knn.predict_proba([exp])[0]
        if pred == 0:
            st.success( f"{pred} -- no default " )
            st.success( pred_proba[pred] )
        elif pred != 0:
            st.error( f"{pred} -- default " )
            st.error( pred_proba[pred] )

if col2.button("Random Forest", disabled = False ):    
    with open("./rfc.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        rfc = load( f )                # загруженная модель rfc
        exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

        pred = rfc.predict([exp])[0]
        pred_proba = rfc.predict_proba([exp])[0]
        if pred == 0:
            st.success( f"{pred} -- no default " )
            st.success( pred_proba[pred] )
        elif pred != 0:
            st.error( f"{pred} -- default " )
            st.error( pred_proba[pred] )

if col3.button("Gradient Boosting", disabled = False ):    
    with open("./boost_gr.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        boost_gr = load( f )                # загруженная модель boost_gr
        exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

        pred = boost_gr.predict([exp])[0]
        pred_proba = boost_gr.predict_proba([exp])[0]
        if pred == 0:
            st.success( f"{pred} -- no default " )
            st.success( pred_proba[pred] )
        elif pred != 0:
            st.error( f"{pred} -- default " )
            st.error( pred_proba[pred] )

if col4.button("Decision Tree", disabled = False ):    
    with open("./dtree.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        dtree = load( f )                # загруженная модель dtree
        exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

        pred = dtree.predict([exp])[0]
        pred_proba = dtree.predict_proba([exp])[0]
        if pred == 0:
            st.success( f"{pred} -- no default " )
            st.success( pred_proba[pred] )
        elif pred != 0:
            st.error( f"{pred} -- default " )
            st.error( pred_proba[pred] )                           


    
