import pandas as pd
import numpy as np
import tensorflow as tf
import logging

# Configuração básica de logging para feedback visual profissional no terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SalesDataPipeline:
    """Classe responsável pelo carregamento, transformação e análise básica dos dados."""
    
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        self.df = None

    def load_and_transform(self) -> pd.DataFrame:
        """Converte o dicionário em DataFrame e realiza checagens básicas."""
        logging.info("Iniciando o carregamento dos dados...")
        try:
            self.df = pd.DataFrame(self.data_dict)
            logging.info("Dados carregados com sucesso no formato pandas DataFrame.")
            return self.df
        except Exception as e:
            logging.error(f"Erro ao converter dicionário para DataFrame: {e}")
            raise

    def perform_basic_analysis(self):
        """Imprime uma análise exploratória básica dos dados."""
        if self.df is None or self.df.empty:
            logging.warning("O DataFrame está vazio ou não foi carregado.")
            return

        print("\n" + "="*50)
        print("📊 ANÁLISE BÁSICA DOS DADOS")
        print("="*50)
        
        print("\nVisualização das primeiras linhas (Head):")
        print(self.df.head())
        
        print("\nResumo Estatístico (Describe):")
        print(self.df.describe())
        
        print("\nVerificação de Valores Nulos:")
        print(self.df.isnull().sum())
        print("="*50 + "\n")


class SalesForecaster:
    """Classe responsável pela criação, treinamento e predição utilizando TensorFlow."""
    
    def __init__(self):
        self.model = None
        self.feature_mean = 0
        self.feature_std = 1

    def preprocess_data(self, df: pd.DataFrame, feature_col: str, target_col: str):
        """Normaliza os dados para otimizar o treinamento da rede neural."""
        try:
            X = df[[feature_col]].values
            y = df[target_col].values
            
            # Normalização simples (Z-score) manual para não depender do scikit-learn
            self.feature_mean = np.mean(X)
            self.feature_std = np.std(X)
            
            # Evita divisão por zero
            if self.feature_std == 0:
                self.feature_std = 1e-8
                
            X_scaled = (X - self.feature_mean) / self.feature_std
            return X_scaled, y
        except KeyError as e:
            logging.error(f"Coluna não encontrada no DataFrame: {e}")
            raise

    def build_model(self, input_shape: int = 1):
        """Constrói uma rede neural Sequencial simples para regressão."""
        logging.info("Construindo o modelo TensorFlow...")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)  # Camada de saída para regressão (valor contínuo)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logging.info("Modelo compilado com sucesso (Otimizador: Adam, Loss: MSE).")

    def train(self, X_train, y_train, epochs: int = 200, batch_size: int = 4):
        """Treina o modelo com os dados fornecidos."""
        if self.model is None:
            raise ValueError("O modelo precisa ser construído antes do treinamento. Chame build_model() primeiro.")
            
        logging.info("Iniciando o treinamento do modelo. Isso pode levar alguns segundos...")
        try:
            # Verbose=0 para manter o terminal limpo
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            logging.info("Treinamento concluído!")
            return history
        except Exception as e:
            logging.error(f"Erro durante o treinamento do modelo: {e}")
            raise

    def predict(self, new_features: list) -> list:
        """Faz previsões baseadas em novos dados inseridos."""
        if self.model is None:
            raise ValueError("O modelo não foi treinado/inicializado.")
            
        try:
            # Aplica a mesma normalização usada no treinamento
            X_new = np.array(new_features).reshape(-1, 1)
            X_new_scaled = (X_new - self.feature_mean) / self.feature_std
            
            predictions = self.model.predict(X_new_scaled, verbose=0)
            return predictions.flatten().tolist()
        except Exception as e:
            logging.error(f"Erro ao realizar a previsão: {e}")
            raise


def main():
    # 1. Dataset em formato de dicionário Python (Dados de Exemplo)
    # Relação: Investimento em Marketing x Vendas Realizadas
    raw_sales_data = {
        'mes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'investimento_marketing': [1000, 1200, 1500, 1300, 1600, 2000, 2200, 2500, 2300, 2800, 3000, 3500],
        'vendas': [5000, 5500, 6200, 5800, 6500, 7500, 8000, 8800, 8200, 9500, 10500, 12000]
    }

    try:
        # 2. Instanciação e Processamento de Dados
        pipeline = SalesDataPipeline(raw_sales_data)
        df_sales = pipeline.load_and_transform()
        pipeline.perform_basic_analysis()

        # 3. Preparação para o Modelo de Machine Learning
        forecaster = SalesForecaster()
        
        # Separando features (variável independente) e target (variável dependente)
        X_train, y_train = forecaster.preprocess_data(
            df=df_sales, 
            feature_col='investimento_marketing', 
            target_col='vendas'
        )

        # 4. Construção e Treinamento
        forecaster.build_model(input_shape=1)
        forecaster.train(X_train, y_train, epochs=300)

        # 5. Realizando Previsões de Teste
        novos_investimentos = [3200, 4000, 5000]
        logging.info(f"Gerando previsões para os seguintes investimentos: {novos_investimentos}")
        
        previsoes = forecaster.predict(novos_investimentos)

        # 6. Exibição dos Resultados
        print("\n" + "="*50)
        print("🎯 RESULTADOS DA PREVISÃO")
        print("="*50)
        for inv, prev in zip(novos_investimentos, previsoes):
            print(f"Investimento R$ {inv:.2f}  -->  Previsão de Vendas: R$ {prev:.2f}")
        print("="*50 + "\n")

    except Exception as e:
        logging.error(f"A execução do pipeline falhou de forma inesperada: {e}")

if __name__ == "__main__":
    # Desativa avisos do TensorFlow para manter a saída limpa
    tf.get_logger().setLevel('ERROR')
    main()