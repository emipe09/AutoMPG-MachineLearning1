import pandas as pd
import time

from funcoes import *
from ucimlrepo import fetch_ucirepo 

print(tf.__version__)

# Buscando o conjunto de dados
auto_mpg = fetch_ucirepo(id=9) 

# Transformando em data frame
data = pd.DataFrame(auto_mpg.data.features, columns=auto_mpg.data.feature_name)
# Adicionando mpg ao data frame
data['mpg'] = auto_mpg.data.targets
# Visualizando os dados
print(data.head())
# Estatísticas dos dados
print(data.describe().transpose())

# Limpando os Dados
data = data.dropna()
# Convertendo a coluna 'origin' em one-hot encoding (3 colunas binarias para indicar origem)
origin = data.pop('origin')
data['EUA'] = (origin == 1) * 1.0
data['EUA'] = (origin == 1) * 1.0
data['Europa'] = (origin == 2) * 1.0
data['Japão'] = (origin == 3) * 1.0

# Separando dados de treino e teste, pegando a variável alvo 'mpg' e normalizando
train_data, test_data, train_labels, test_labels = preprocess_data(data)

# Criando e compilando o modelo
model = model_maker(train_data)
# Treinando o modelo
history = model_training(model, train_data, train_labels, 0.2)

# Plotando o histórico de treinamento
plot_training(history)

# Avaliando o modelo
loss, mae, mse = model.evaluate(test_data, test_labels, verbose=2)
accuracy = (1 - mae / test_labels.mean()) * 100
print("\nAcurácia do Modelo: {:5.2f}%".format(accuracy))
print('\nErro médio absoluto do MPG:  {:5.2f}+/-'.format(mae))

# Fazendo predições
test_predictions = model.predict(test_data).flatten()
# Plotando a previsão com a realidade
plot_predictions(test_labels, test_predictions)

# Erro de distribuição
plot_error_distribution(test_predictions, test_labels)








