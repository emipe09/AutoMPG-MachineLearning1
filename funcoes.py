import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import time

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Dividindo os dados em treino e teste
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)

    # A variável alvo é 'mpg' e vamos separá-la:
    train_labels = train_data.pop('mpg')
    test_labels = test_data.pop('mpg')

    # Normalizando os dados
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data, train_labels, test_labels

def model_maker(train_data):
    # Construindo o modelo
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_data[0])]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # Compilando o modelo
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        metrics=['mae', 'mse']
    )
    
    return model

def model_training(model, train_data, train_labels, validation_split):
    epochs = 1000
    # Essa variavel é usada para parar o treinamento quando o erro de validação não melhorar mais
    quality_control = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    start = time.time()
    history = model.fit(
        train_data, train_labels,
        epochs=epochs, validation_split=validation_split, verbose=0,
        callbacks=[quality_control])
    end = time.time()
    print("Tempo de treinamento: ", end - start)
    return history


def plot_training(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Erro médio absoluto [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
        label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
        label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Erro médio quadrático [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
        label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
        label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

def plot_predictions(test_labels, test_predictions):

    # Plotando a previsão com a realidade
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('Valores reais [MPG]')
    plt.ylabel('Previsões [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

def plot_error_distribution(predictions, labels):
    error = predictions - labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    plt.ylabel("Count")
    plt.show()





