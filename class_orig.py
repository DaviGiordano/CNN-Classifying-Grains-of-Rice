"""
Aluno: Davi Giordano Valério 11805273

Esse script classifica imagens de arroz não alinhadas, de formato .jpg
No diretório raiz deve conter uma pasta com a estrutura:

data
    Fold 1
    Fold 2
    Fold 3
    Fold 4
    Fold 5

Em que cada pasta contém os dados para o fold correspondente especificado no enunciado.
Com isso, o script treina cinco modelos diferentes, usando 5-fold cross-validation, imprime a acurácia de cada modelo,
a acurácia média e o desvio padrão
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder

def load_images_and_labels(directories, exclude_dir=None):
    """
    Função que importa imagens de uma lista de diretórios 'directories' e exlcui o diretório 'exclude_dir'
    Importa todos os arquivos do tipo .jpg
    Atribui um label correspondente ao texto antes do espaço no nome do arquivo
    """
    labels = []
    images = []
    for directory in directories:
        for dir_name, _, file_list in os.walk(directory):
            if dir_name == exclude_dir:  # Pula o diretório a exlcuir
                continue
            for filename in file_list:
                if filename.endswith('.jpg'):
                    # A label é o nome do arquivo antes do espaço
                    label = filename.split(' ')[0]
                    labels.append(label)

                    # Importa a imagem
                    img_path = os.path.join(dir_name, filename)

                    # Converte a imagem para 32x32
                    img = load_img(img_path, target_size=(32, 32))  # the target size depends on your requirements
                    
                    # Transforma a imagem para valores de 0 a 1
                    img_array = img_to_array(img) / 255.0
                    images.append(img_array)
    
    print(f"Número de imagens: {len(images)}, Número de labels: {len(labels)}")

    if not images:
        print(f"Imagens não encontradas em: {directories}")

    # Converte as labels para one-hot-encoding
    le = LabelEncoder()
    if labels:
        labels = le.fit_transform(labels)
        labels = tf.keras.utils.to_categorical(labels)
    else:
        print("Labels não encontradas")
        return [], []

    return images, labels


def build_model():
    """
    Estrutura modelo convolucional do tipo Lenet-5 para classificação do arroz
    """
    model = Sequential()

    # Camada 1: Convolução + Average Pooling
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 3)))
    model.add(AveragePooling2D())

    # Camada 2: Convolução + Average Pooling
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))
    model.add(AveragePooling2D())

    # Transformando em linha de valores
    model.add(Flatten())

    # Camada 3: Camada densa
    model.add(Dense(units=120, activation='tanh'))

    # Camada 4: Camada densa
    model.add(Dense(units=84, activation='tanh'))

    # Camada 5: Cama de saída
    model.add(Dense(units=5, activation = 'softmax'))

    return model


# Array para guardar a acurácia dos diferentes folds
models_accuracy = []


from tensorflow.keras.optimizers import Adam

# Parâmetros de treinamento
batch_size = 32
epochs = 20

def train_model(train_images, train_labels, verbose):
    """
    Recebe imagens de treino e seus labels correspondentes. 
    Verbose escolhe a quantidade de texto imprimir durante o treinamento
    Retorna o modelo treinado com as imagens especificadas
    """
    
    # Building model
    model= build_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Converte imagens para array
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Treina o modelo
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)
    return model


def test_model(model, test_images, test_labels):
    """
    Recebe um modelo, imagens de teste e seus labels esperados
    Retorna a acurácia obtida para esse modelo com esse conjunto de teste
    """
    
    # Converte as imagens para array
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Testa o modelo
    loss, accuracy = model.evaluate(test_images, test_labels)

    print(f"Acurácia para dados de teste: {accuracy * 100:.2f}%")
    return accuracy


## Fold 1 ##

print("\n")
train_dirs = ["./data/Fold1", "./data/Fold2", "./data/Fold3", "./data/Fold4"]
test_dir = "./data/Fold5"

train_images, train_labels = load_images_and_labels(train_dirs, exclude_dir=test_dir)
test_images, test_labels = load_images_and_labels([test_dir])

model = train_model(train_images, train_labels, verbose=0)

accuracy = test_model(model, test_images, test_labels)
models_accuracy.append(accuracy)

## Fold 2 ##

print("\n")
train_dirs = ["./data/Fold5", "./data/Fold1", "./data/Fold2", "./data/Fold3"]
test_dir = "./data/Fold4"

train_images, train_labels = load_images_and_labels(train_dirs, exclude_dir=test_dir)
test_images, test_labels = load_images_and_labels([test_dir])

model = train_model(train_images, train_labels, verbose=0)

accuracy = test_model(model, test_images, test_labels)
models_accuracy.append(accuracy)


## Fold 3 ##

print("\n")
train_dirs = ["./data/Fold4", "./data/Fold5", "./data/Fold1", "./data/Fold2"]
test_dir = "./data/Fold3"

train_images, train_labels = load_images_and_labels(train_dirs, exclude_dir=test_dir)
test_images, test_labels = load_images_and_labels([test_dir])

model = train_model(train_images, train_labels, verbose=0)

accuracy = test_model(model, test_images, test_labels)
models_accuracy.append(accuracy)

## Fold 4 ##

print("\n")
train_dirs = ["./data/Fold3", "./data/Fold4", "./data/Fold5", "./data/Fold1"]
test_dir = "./data/Fold2"

train_images, train_labels = load_images_and_labels(train_dirs, exclude_dir=test_dir)
test_images, test_labels = load_images_and_labels([test_dir])

model = train_model(train_images, train_labels, verbose=0)

accuracy = test_model(model, test_images, test_labels)
models_accuracy.append(accuracy)

## Fold 5 ##

print("\n")
train_dirs = ["./data/Fold2", "./data/Fold3", "./data/Fold4", "./data/Fold5"]
test_dir = "./data/Fold1"

train_images, train_labels = load_images_and_labels(train_dirs, exclude_dir=test_dir)
test_images, test_labels = load_images_and_labels([test_dir])

model = train_model(train_images, train_labels, verbose=0)

accuracy = test_model(model, test_images, test_labels)
models_accuracy.append(accuracy)

## Análise dos Resultados ##

print("\n")
mean_accuracy = np.average(models_accuracy)
std_accuracy = np.std(models_accuracy)
print(f"Acurácia média: {mean_accuracy * 100:.2f}%")
print(f"Desvio Padrão: {std_accuracy* 100:.2f}%")



