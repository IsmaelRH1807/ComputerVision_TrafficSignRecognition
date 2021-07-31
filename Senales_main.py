import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

################# Parameters #####################

path = "myData"  # Ruta de la carpeta con las imágenes
labelFile = 'labels.csv'  # Nombre del csv que tiene los labels de las clases
batch_size_val = 15  # Número de datos que se van a procesar en un batch
steps_per_epoch_val = 350 # Número de iteraciones por lotes antes de que una epoch de entrenamiento se considere finalizada
epochs_val = 8 # Número de veces que se ejecutarán los algoritmos de aprendizaje con todos los datos.
dimensionesImagen = (32, 32, 3) # Dimensión de la imagen, en este caso es 32x32x3, las dimensiones se convertiran en el input (capa de entrada)
testRatio = 0.2  # Porcentaje de imágenes que serán para el test
validationRatio = 0.2  # Porcentaje de imágenes de las que quedan que serán para la validación

######################## Importar las imágenes
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Número de clases detectadas:", len(myList))
noOfClasses = len(myList)
for x in range(0, len(myList)):     # for con el número de clases
    listaImagenes = os.listdir(path + "/" + str(count))
    for y in listaImagenes:         #for que recorre cada imagen
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)      #utilizo imread para leer la imagen
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)   # hago un array con las imágenes
classNo = np.array(classNo)     # hago un array de las clases

############################### Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio) # divido los arrays de imágenes y clases en paquetes random para entrenamiento y testeo
y_test_MC = y_test                                                                                          # para eso uso el porcentaje que definimos al principio
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio) # divido los arrays de imágenes y clases en paquetes random para entrenamiento y validación
y_val_MC = y_validation                                                                                          # igual utilizo el porcetaje para los datos de validación que definimos al principio

############################### Verificar que el número de imágenes coincida con el número de labels en cada dataset
print("Data Shapes")
print("Entrenamiento", end=" ");
print("X: ", X_train.shape[0], "Y: ", y_train.shape[0])
print("Testeo", end=" ");
print("X: ", X_test.shape[0], "Y: ", y_test.shape[0])
print("Validacion", end=" ");
print("X: ", X_validation.shape[0], "Y: ", y_validation.shape[0])
data = pd.read_csv(labelFile)   #leemos los labels del csv

# Preprocesamiento de las imágenes

def escala_grises(img):         # convertir la imagen a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):          # Equalizar la imagen (equilibrar la iluminación o luminosidad de la imagen)
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = escala_grises(img)
    img = equalize(img)
    img = img / 255  # se divide para 255 para normalizar los datos de 0 a 1
    return img


X_train = np.array(list(map(preprocessing, X_train)))  # Preprocesamos todas las imágenes y las guardamos en la misma variable X_train
X_validation = np.array(list(map(preprocessing, X_validation))) # Lo mismo para los datos de validación
X_test = np.array(list(map(preprocessing, X_test))) #Lo mismo para los datos de testeo
cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])  # TO CHECK IF THE TRAINING IS DONE PROPERLY
X_test_MC = X_test


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # aplanos las imágenes a un array de 1 dimensión
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


# Esto me permite crear nuevas imágenes a partir de mi data set mediante transformaciones
dataGen = ImageDataGenerator(width_shift_range=0.1, # me permite desplazar randómicamente la imagen en un rango del 10% de su ancho
                             height_shift_range=0.1, # me permite desplazar randómicamente la imagen en un rango del 10% de su alto
                             zoom_range=0.2,  # me permite hacer zoom a la imagen en un rango del 20%
                             shear_range=0.1,  # me permite desplazar las esquinas de la imagen
                             rotation_range=10)  # grados en los que deseo rotar la imagen

dataGen.fit(X_train) # ajusto el generador a mis datos de entrenamiento
batches = dataGen.flow(X_train, y_train, batch_size=20)  # Genero batches de imágenes aumentadas
X_batch, y_batch = next(batches)

##
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.tight_layout()
from PIL import Image
for i in range(4):
    img = Image.open('samples/sample_random'+str(i)+'.png')
    axs[i].imshow(img)
    axs[i].axis('off')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.tight_layout()
for i in range(4):
    img = cv2.imread('samples/sample_random'+str(i)+'.png')
    axs[i].hist(img.flatten(), 256, [0, 256])
plt.show()

#Histogramas de la data
img_prep = cv2.imread('samples/sample_random0.png')
plt.hist(img_prep.flatten(), 256, [0, 256])
plt.show()

img_prep = cv2.imread('samples/sample_random1.png')
plt.hist(img_prep.flatten(), 256, [0, 256])
plt.show()

img_prep = cv2.imread('samples/sample_random2.png')
plt.hist(img_prep.flatten(), 256, [0, 256])
plt.show()

img_prep = cv2.imread('samples/sample_random3.png')
plt.hist(img_prep.flatten(), 256, [0, 256])
plt.show()

# Mostrar imágenes aumentadas
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(dimensionesImagen[0], dimensionesImagen[1]))
    axs[i].axis('off')
plt.show()

# transformo los índices de las categorías a una matriz binaria
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


############################### Definición del modelo
def myModel():
    num_Filtros = 60
    tamanio_Filtro = (5, 5)  # Valor que se mueve por la imagen para extraer las carácterísticas
    tamanio2_Filtro = (3, 3)
    size_of_pool = (2, 2)
    numNodos = 500  # Número de nodos de las capas intermedias
    model = Sequential() # Instancio un modelo tipo Sequential()
    model.add((Conv2D(num_Filtros, tamanio_Filtro, input_shape=(dimensionesImagen[0], dimensionesImagen[1], 1), activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(num_Filtros, tamanio_Filtro, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  # Submuestreo

    model.add((Conv2D(num_Filtros // 2, tamanio2_Filtro, activation='relu')))
    model.add((Conv2D(num_Filtros // 2, tamanio2_Filtro, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten()) # Utilizo Flatten() para convertir la matriz de características de 3D a 1D
    model.add(Dense(numNodos, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))  # Capa de salida
    # Compilamos el modelo
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

############################### TRAIN
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val, epochs=epochs_val, validation_data=(X_validation, y_validation), shuffle=1)

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Guardamos el modelo entrenado
model.save("model_trained.h5")

#Saco mi matriz de confusión

predicted_classes = model.predict_classes(X_validation)
cm = confusion_matrix(y_val_MC, predicted_classes)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


predicted_classes = model.predict_classes(X_test)
cm = confusion_matrix(y_test_MC, predicted_classes)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
cv2.waitKey(0)
