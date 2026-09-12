from sklearn.tree import DecisionTreeClassifier #CLassificação entre os dados inseridos
import numpy as np #Bilbioteca para calculos extensos 
import streamlit as st #Biblioteca para mostrar o resultado

# Tempo de Uso de Produto X numero de Reclamações

x = np.array([
    [1,1],
    [1,5],
    [5,4],
    [3,3],
    [4,1],
    [5,0],
])


# 0 - > fica - 1 -> cancela

y = np.array([0,1,1,0,1,1])

modelo = DecisionTreeClassifier()
modelo.fit(x,y)

uso = st.number_input("Quantidade de Vezes que o Produto Foi utilizado:  ", value = 0)
reclamacoes = st.number_input("Reclamações: ", value = 0)

st.header ("Análise De Cancelamento")
if st.button('Analisar Cliente'):
    if modelo.predict([[uso,reclamacoes]]) == 1:
        st.write("Cliente Consolidado")
else:
    st.write("Possivel Cancelamento")

print(modelo.predict([[5,2]]))