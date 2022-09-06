import xgboost as xgb
import streamlit as st
import pandas as pd

# carregando o modelo de regressão linear que foi criado.
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

# Cache do modelo para carregamento mais rápido.
@st.cache


# Função que recebe a entrada dos usuários(características do diamante) e gera uma previsão de preço.
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Predicting the price of the carat
    if cut == 'Fair':
        cut = 0
    elif cut == 'Good':
        cut = 1
    elif cut == 'Very Good':
        cut = 2
    elif cut == 'Premium':
        cut = 3
    elif cut == 'Ideal':
        cut = 4

    if color == 'J':
        color = 0
    elif color == 'I':
        color = 1
    elif color == 'H':
        color = 2
    elif color == 'G':
        color = 3
    elif color == 'F':
        color = 4
    elif color == 'E':
        color = 5
    elif color == 'D':
        color = 6

    if clarity == 'I1':
        clarity = 0
    elif clarity == 'SI2':
        clarity = 1
    elif clarity == 'SI1':
        clarity = 2
    elif clarity == 'VS2':
        clarity = 3
    elif clarity == 'VS1':
        clarity = 4
    elif clarity == 'VVS2':
        clarity = 5
    elif clarity == 'VVS1':
        clarity = 6
    elif clarity == 'IF':
        clarity = 7

    prediction = model.predict(pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']))
    return prediction

# Aplicativo e áreas para entradas do usuário.
st.title('Previsor de preços de diamantes.')
st.image("""diamente.jpg""", caption=None, width=None)
st.header('Informe as características do diamante')

# cria um conjunto de campos para entrada das informações.
carat = st.number_input('Peso em quilates:', min_value=0.1, max_value=10.0, value=1.0)
cut = st.selectbox('Classificação de corte:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Classificação de cores:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Classificação de clareza:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Porcentagem de profundidade do diamante:', min_value=0.1, max_value=100.0, value=1.0)
table = st.number_input('Porcentagem da Mesa Diamante:', min_value=0.1, max_value=100.0, value=1.0)
x = st.number_input('Comprimento do diamante (X) em mm:', min_value=0.1, max_value=100.0, value=1.0)
y = st.number_input('Largura do Diamante (Y) em mm:', min_value=0.1, max_value=100.0, value=1.0)
z = st.number_input('Altura do diamante (Z) em mm:', min_value=0.1, max_value=100.0, value=1.0)

if st.button('Prever Preço'):
    price = predict(carat, cut, color, clarity, depth, table, x, y, z)
    st.success(f'O preço previsto do diamante é ${price[0]:.2f} USD')

st.write("@author: Juliano Sarnes Longo")
st.write("SET/2022")

