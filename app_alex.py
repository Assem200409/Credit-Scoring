import streamlit as st
#import pandas as pd
#import numpy as np
from pickle import load, loads
#from assem.class_preparing_5_ import Preparing
#from sklearn.neighbors import KNeighborsClassifier # Импорт модели

st.title("Credit Scoring in Taiwan 2005")

my_data = st.sidebar.selectbox("Choose your dataset",
                 ["UCI_Credit_Card.csv"]
                  )
#shold = st.slider('shold',min_value=0, max_value=1, step=0.01, format=None)
# if st.button("Load", disabled = False ):
#     df = pd.read_csv( my_data )
#     prep = Preparing(df)                # в переменной prep создаем экземпляр класса Preparing
#     prep.warning()                      # переименовываем колонку на "default"
#     prep.rename_columns()               # выгружаем переименованный датасет
#     prep.select_columns()
#     prep.group_clients()
#     X_all_new, y_new = prep.X_all_new, prep.y_new
#     st.write('ok')
#     with open("./model.pkl", "rb") as model_pkl: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
#         knn = load( model_pkl )           # загруженная модель knn
#         #knn = KNeighborsClassifier( n_neighbors = 51, p = 1, weights = 'distance' ) # Создаем модель, n_neighbors = 5 -параметр
#         #knn.fit( X, y ) # Тренировка модели

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


#st.write( pd1 )
st.success( f"credit = {limit_ball}" )

st.title("Predictions")

#col1, col2 = st.columns[2]

#col1.success( limit_ball )
#col2.success( limit_ball )



uploaded_file = st.file_uploader( "My models" )

if uploaded_file is not None:
    model_bytes = uploaded_file.read()
    boost_gr = loads(model_bytes) #boost_gr.pkl
    
    # pkl
    # with open("./boost_gr.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
    #     model_bytes = load( f )           # загруженная модель knn
    # boost_gr = loads( model_bytes )
    st.write("Model loaded successfully")
    exp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

    #st.write("Prediction", loaded_clf.predict([exp]))

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     # To read file as bytes:
#     bytes_data = uploaded_file.getvalue()
#     st.write(bytes_data)


#if st.button("KNN prediction", disabled = False ):
# if st.button("Random Forest", disabled = False ):    
#     with open("./model.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
#         rfc = load( f, encoding='ASCII' )           # загруженная модель knn
#         #exp = np.array([limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6])
#         exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

#         pred = rfc.predict([exp])[0]
#         pred_proba = rfc.predict_proba([exp])[0]
#         if pred == 0:
#             st.success( f"{pred} -- no default " )
#             #st.success( f"probability = {round(pred_proba,2)}" )
#             st.success( pred_proba[pred] )
#         elif pred != 0:
#             st.error( f"{pred} -- default " )
#             #st.warning( f"probability = {round(pred_proba,2)}" )
#             st.error( pred_proba[pred] )

if st.button("Gradient Boosting", disabled = False ):    
    with open("./boost_gr.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
        #boost_gr = load( f )           # загруженная модель knn
        #exp = np.array([limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6])
        exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

        pred = boost_gr.predict([exp])[0]
        pred_proba = boost_gr.predict_proba([exp])[0]
        if pred == 0:
            st.success( f"{pred} -- no default " )
            #st.success( f"probability = {round(pred_proba,2)}" )
            st.success( pred_proba[pred] )
        elif pred != 0:
            st.error( f"{pred} -- default " )
            #st.warning( f"probability = {round(pred_proba,2)}" )
            st.error( pred_proba[pred] )

# if st.button("Decision Tree", disabled = False ):    
#     with open("./dtree.pkl", "rb") as f: # rb - режим чтения, ./model.pkl - файл куда сохраняется модель
#         dtree = load( f )           # загруженная модель knn
#         #exp = np.array([limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6])
#         exp = [limit_ball, pd1, pd2, pd3, pd4, pd5, pd6, b1, b2, b3, b4, b5, b6, p1, p2, p3, p4, p5, p6]

#         pred = dtree.predict([exp])[0]
#         pred_proba = dtree.predict_proba([exp])[0]
#         if pred == 0:
#             st.success( f"{pred} -- no default " )
#             #st.success( f"probability = {round(pred_proba,2)}" )
#             st.success( pred_proba[pred] )
#         elif pred != 0:
#             st.error( f"{pred} -- default " )
#             #st.warning( f"probability = {round(pred_proba,2)}" )
#             st.error( pred_proba[pred] )                           


    
