import streamlit as st 
import pandas as pd     
import numpy as np     
from assem.class_preparing_5_ import Preparing 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('UCI_Credit_Card.csv')

st.title("Credit Scoring in Taiwan 2005")

my_data = st.sidebar.selectbox("Choose your dataset",
                 ["UCI_Credit_Card.csv"]
                  )
probas = [round(0.01*p,2) for p in range(10,91)]
shold = st.sidebar.selectbox("Choose your shold-probability",
                 probas
                  )
@st.cache_data
def load_dataset(path = 'UCI_Credit_Card.csv'):
     df = pd.read_csv(path)
     prep = Preparing(df)                # в переменной prep создаем экземпляр класса Preparing
     prep.warning()                      # переименовываем колонку на "default"
     prep.rename_columns()               # выгружаем переименованный датасет
     prep.select_columns()
     prep.group_clients()
     return prep.X_all_new, prep.y_new

X, y = load_dataset()

X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size = 0.2, random_state = 0  ) # split разбиение данных

@st.cache_resource
def train_knn(X_train, y_train):
    knn = KNeighborsClassifier( n_neighbors = 101, p = 1, weights = 'distance' )
    knn.fit(X_train, y_train)
    return knn

knn = train_knn(X_train, Y_train) 
with st.form('Input:'):
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

    submitted = st.form_submit_button("Submit ") 

st.success( f"credit = {limit_ball}" )
st.title("Predictions")

if st.button("KNN prediction", disabled = False ): 
        #exp = np.array([limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6])
        exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

        pred = knn.predict([exp])[0]
        pred_proba = knn.predict_proba([exp])[0]
        if pred == 0:
            st.success( f"{pred} -- no default " )
            st.success( pred_proba[pred] )
        elif pred != 0:
            st.error( f"{pred} -- default " )
            st.error( pred_proba[pred] )
