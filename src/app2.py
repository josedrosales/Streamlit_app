import streamlit as st
import pickle
from pickle import load

# Cargar el modelo entrenado
with open('../models/iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

#Diccionario de clases
class_dict = {
    "0": "iris setosa",
    "1": "iris versicolor",
    "2": "iris virginica"
}

# Título de la aplicación
st.title('Modelo de Predicción de Iris')

# Entrada de datos del usuario para hacer predicciones
#st.sidebar.header('Parámetros de entrada')
sepal_length = st.slider('Sepal length (cm)', min_value=0.0, max_value=6.0, step=0.1)
sepal_width = st.slider('Sepal width (cm)', min_value=0.0, max_value=6.0, step=0.1)
petal_length = st.slider('Petal length (cm)', min_value=0.0, max_value=6.0, step=0.1)
petal_width = st.slider('Petal width (cm)', min_value=0.0, max_value=6.0, step=0.1)
#predict_df=pd.DataFrame({"a":[sepal_length],'b':[sepal_width],'c':[petal_length],'d':[petal_width]})
# Predicción basada en los parámetros de entrada del usuario
if st.button("Predict"):
    prediction = str(model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:",pred_class)