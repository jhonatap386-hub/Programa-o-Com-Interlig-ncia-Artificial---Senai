import streamlit as st
import pandas as pd

# ProCoding -> não Utiliza a IA Generativa

dados = pd.read_csv("Vendas.csv")

st.header("Calculadora Streamlit")
st.write("Adicione os números para calcular: ")

n1 = st.number_input("Digite um Número: ", min_value = 0)
n2 = st.number_input("Digite outro Número: ", min_value = 0)

soma_, div_, sub_, mult_ = st.columns(4)

if soma_.button("+"):
    soma = n1 + n2
    st.info(soma)
elif sub_.button("-"):
    sub = n1 - n2
    st.info(sub)
elif mult_.button("x"):
    mult = n1 * n2
    st.info(mult)
elif div_.button("/"):
    div = n1 / n2
    st.info(div)

if st.button("Mostrar Mapa"):
    st.map()

if st.button("Análise de Dados"):
    st.table(dados)
    st.bar_chart(dados, x = "ano", y = "Lucro")
    st.scatter_chart(dados, x = "venda",  y = "Lucro")
    st.line_chart(dados, x = "ano", y = "venda")


### Outra Aba com Csv ###

#ano,venda,Lucro
#1998,5000.0,2500.6
#1999,4500.5,1500.5
#2000,2000.0,3000.3
#2021,3500.0,800.9
#2023,4500.5,1500.6
#2024,4500.02,2500.
#2025,5000,950.9
#2026,7000.60,1500.0
