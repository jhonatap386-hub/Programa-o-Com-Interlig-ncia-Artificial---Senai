## Atividade 1 / Nota dos Alunos ##

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Preparação dos Dados
# Criamos o DataFrame preenchendo os dados corretamente
estudos = pd.DataFrame(
    {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
)

# O TensorFlow trabalha melhor com arrays do NumPy do tipo float32
X = estudos["horas"].values.astype(np.float32)  # Característica (Feature)
y = estudos["notas"].values.astype(np.float32)  # Alvo (Target)

# 2. Construção da Arquitetura do Modelo
# Adicionado o parâmetro correto: input_shape=[1] (pois passamos 1 número por vez)
model = models.Sequential([layers.Dense(units=1, input_shape=[1])])

# 3. Compilação do Modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="mean_squared_error",
)

# 4. Treinamento (Fit)
print("Iniciando o treinamento...")
history = model.fit(
    X, y, epochs=500, verbose=0
)  # Oculta o progresso linha por linha
print("Treinamento concluído com sucesso!")

# 5. Predição
# Vamos prever as notas para quem estuda 6 e 8 horas
horas_teste = np.array([6.0, 8.0], dtype=np.float32)
predicoes = model.predict(horas_teste)

# Exibindo os resultados
print("\n--- Resultados das Predições ---")
for h, p in zip(horas_teste, predicoes):
    print(
        f"Com {h} horas de estudo, a nota prevista é aproximadamente: {p[0]:.2f}"
    )


## Atividade 2 / Sono Gamer ##

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Preparação dos Dados
# Criamos o DataFrame com a relação de horas de jogo e cansaço
gamer = pd.DataFrame(
    {
        "horas_jogo":,
        "cansaco":,
    }
)

# Convertemos as colunas para arrays do NumPy do tipo float32 (padrão do TensorFlow)
X = gamer["horas_jogo"].values.astype(np.float32)  # Característica (Feature)
y = gamer["cansaco"].values.astype(np.float32)  # Alvo (Target)

# 2. Construção da Arquitetura do Modelo
# Criamos uma rede neural simples com 1 neurônio e 1 entrada
model = models.Sequential(
    [
        layers.Dense(
            units=1, input_shape=[1]
        )  # units=1 (1 neurônio), input_shape=[1] (1 entrada: horas)
    ]
)

# 3. Compilação do Modelo
# Definimos o otimizador Adam para ajustar os pesos e o MSE como função de perda
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.1
    ),  # Taxa de aprendizado inicial
    loss="mean_squared_error",  # Erro Quadrático Médio
)

# 4. Treinamento (Fit)
# O modelo vai rodar o conjunto de dados por 500 épocas para aprender a relação
print("Iniciando o treinamento do modelo...")
history = model.fit(
    X, y, epochs=500, verbose=0
)  # verbose=0 oculta as linhas de log por época
print("Treinamento concluído com sucesso!")

# 5. Predição
# Testando o modelo: Qual seria o nível de cansaço para quem joga 5 ou 7 horas?
horas_teste = np.array([5.0, 7.0], dtype=np.float32)
predicoes = model.predict(horas_teste)

# Exibindo os resultados finais no terminal
print("\n--- Resultados das Predições ---")
for horas, previsao in zip(horas_teste, predicoes):
    print(
        f"Jogando por {horas} horas, o nível de cansaço previsto é: {previsao[0]:.2f}"
    )

## Atividade 3 / IA do Sorvete ##

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Preparação dos Dados
# Criamos o DataFrame relacionando a temperatura com as vendas de sorvete
sorvete = pd.DataFrame(
    {
        "temperatura":,
        "vendas":,
    }
)

# Convertemos os dados para arrays do NumPy do tipo float32 (padrão aceito pelo TensorFlow)
X = sorvete["temperatura"].values.astype(np.float32)  # Característica (Feature)
y = sorvete["vendas"].values.astype(np.float32)  # Alvo (Target)

# 2. Construção da Arquitetura do Modelo
# Criamos um modelo sequencial contendo 1 neurônio com 1 entrada (temperatura)
model = models.Sequential(
    [
        layers.Dense(
            units=1, input_shape=
        )  # 1 neurônio simula perfeitamente uma regressão linear
    ]
)

# 3. Compilação do Modelo
# Configuramos o algoritmo de ajuste (Adam) e a métrica de erro (MSE)
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.1
    ),  # Taxa de aprendizado inicial
    loss="mean_squared_error",  # Erro Quadrático Médio
)

# 4. Treinamento (Fit)
# O modelo passará pelos dados por 1000 épocas para ajustar os parâmetros matemáticos
print("Iniciando o treinamento do modelo no TensorFlow...")
history = model.fit(
    X, y, epochs=1000, verbose=0
)  # verbose=0 silencia o log de cada época
print("Treinamento concluído com sucesso!")

# 5. Predição
# Vamos prever as vendas para dias com temperaturas de 22°C e 32°C
temperaturas_teste = np.array([22.0, 32.0], dtype=np.float32)
predicoes = model.predict(temperaturas_teste)

# Exibindo as previsões textuais no terminal
print("\n--- Resultados das Predições ---")
for temp, prev in zip(temperaturas_teste, predicoes):
    print(
        f"Com a temperatura de {temp}°C, a previsão de vendas é de: {prev:.1f} sorvetes"
    )


## Atividade 4 / Detector de Aprovação Ninja ##


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Preparação dos Dados
# Criamos o DataFrame relacionando o número de faltas ao resultado (1 = Aprovado, 0 = Reprovado)
alunos = pd.DataFrame({"faltas":, "resultado": [1, 1, 1, 0, 0, 0]})

# Convertemos os dados para arrays do NumPy do tipo float32
X = alunos["faltas"].values.astype(np.float32)  # Característica (Feature)
y = alunos["resultado"].values.astype(np.float32)  # Alvo/Classe (Target)

# 2. Construção da Arquitetura do Modelo
# Para classificação binária, adicionamos a função de ativação 'sigmoid' no neurônio
model = models.Sequential(
    [
        layers.Dense(
            units=1,
            input_shape=[1],
            activation="sigmoid",  # Transforma a saída em uma probabilidade entre 0 e 1
        )
    ]
)

# 3. Compilação do Modelo
# Mudamos a função de perda para 'binary_crossentropy', ideal para classificar 0 ou 1
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",  # Entropia Cruzada Binária
    metrics=["accuracy"],  # Métrica para acompanhar a porcentagem de acertos
)

# 4. Treinamento (Fit)
# O modelo passará pelos dados por 800 épocas para aprender a linha de decisão
print("Iniciando o treinamento do classificador no TensorFlow...")
history = model.fit(
    X, y, epochs=800, verbose=0
)  # verbose=0 silencia o log de cada época
print("Treinamento concluído com sucesso!")

# 5. Predição / Teste
# Vamos testar o modelo com alunos que tiveram 1 falta, 4 faltas e 8 faltas
faltas_teste = np.array([1.0, 4.0, 8.0], dtype=np.float32)
probabilidades = model.predict(faltas_teste)

# Exibindo os resultados textuais no terminal com base no limiar de 50% (0.5)
print("\n--- Resultados da Classificação ---")
for faltas, prob in zip(faltas_teste, probabilidades):
    # O retorno da ativação sigmoid é a probabilidade (ex: 0.85 = 85% de chance de ser 1)
    status = "APROVADO" if prob >= 0.5 else "REPROVADO"
    print(
        f"Aluno com {faltas:.0f} falta(s) -> Probabilidade de aprovação: {prob[0]*100:.1f}% -> Classificação: {status}"
    )


## Atividade 5 / IA do Pet Feliz ## 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Preparação dos Dados
# Criamos o DataFrame relacionando a quantidade de passeios com a felicidade do cachorro
pets = pd.DataFrame({"passeios":, "felicidade": [2, 4, 5, 8, 10]})

# Convertemos os dados para arrays do NumPy do tipo float32 (padrão otimizado do TensorFlow)
X = pets["passeios"].values.astype(np.float32)  # Característica (Feature)
y = pets["felicidade"].values.astype(np.float32)  # Alvo (Target)

# 2. Construção da Arquitetura do Modelo
# Criamos um modelo sequencial contendo 1 neurônio com 1 entrada (passeios)
model = models.Sequential(
    [
        layers.Dense(
            units=1, input_shape=[1]
        )  # 1 neurônio reproduz perfeitamente uma regressão linear
    ]
)

# 3. Compilação do Modelo
# Configuramos o otimizador Adam para os ajustes e o MSE como métrica de erro
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.1
    ),  # Taxa de aprendizado inicial
    loss="mean_squared_error",  # Erro Quadrático Médio
)

# 4. Treinamento (Fit)
# O modelo passará pelos dados por 1000 épocas para ajustar os pesos matemáticos
print("Iniciando o treinamento do modelo no TensorFlow...")
history = model.fit(
    X, y, epochs=1000, verbose=0
)  # verbose=0 silencia o log de progresso por época
print("Treinamento concluído com sucesso!")

# 5. Predição
# Vamos prever o nível de felicidade para cães que passeiam 2.5 e 6 vezes
passeios_teste = np.array([2.5, 6.0], dtype=np.float32)
predicoes = model.predict(passeios_teste)

# Exibindo as previsões textuais no terminal
print("\n--- Resultados das Predições ---")
for passeios, previsao in zip(passeios_teste, predicoes):
    print(
        f"Com {passeios} passeio(s), o nível de felicidade previsto do cão é: {previsao[0]:.1f}"
    )

## Atividade 6 / Detector de Filme Bom ##

