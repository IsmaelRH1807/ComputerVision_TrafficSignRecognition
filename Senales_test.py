import numpy as np
import cv2
from keras.models import load_model


#############################################

frameWidth = 640  # Resolución de la cámara
frameHeight = 480
brightness = 180
umbral = 0.85  # Umbral
font = cv2.FONT_HERSHEY_SIMPLEX #Tipo de fuente de la letra

#Ejecutar la cámara

cap = cv2.VideoCapture(0) #0 para la cámara frontal
adress = "http://192.168.1.16:8080/video"
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
cap.open(adress)
# Importar el modelo entrenado
model = load_model('model_trained.h5')

def escala_grises(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = escala_grises(img)
    img = equalize(img)
    img = img / 255
    return img

def nombreClase(classNo):      #Devuelve el nombre de la clase detectada
    if classNo == 0:
        return 'Limite de velocidad (50km/h)'
    elif classNo == 1:
        return 'Limite de velocidad (60km/h)'
    elif classNo == 2:
        return 'Limite de velocidad (80km/h)'
    elif classNo == 3:
        return 'Limite de velocidad (100km/h)'
    elif classNo == 4:
        return 'No rebasar'
    elif classNo == 5:
        return 'Prohibido vehiculos de mas de 3,5 ton'
    elif classNo == 6:
        return 'Ceda el paso'
    elif classNo == 7:
        return 'Stop'
    elif classNo == 8:
        return 'No entre'
    elif classNo == 9:
        return 'Hombres trabajando'

while True:
    # Leemos la imagen capturada
    success, imgOrignal = cap.read()
    resize = cv2.resize(imgOrignal, (1280, 720))

    # Es necesario procesar la imagen antes de evaluarla con el modelo
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)

    #Mostramos cada imagen en la ventana

    img = img.reshape(1, 32, 32, 1)

    #Escribimos en la ventana CLASE
    cv2.putText(imgOrignal, "CLASE: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    #Ahora sí usamos el modelo para predecir la imagen captada
    predicciones = model.predict(img)
    indiceClase = np.argmax(model.predict(img))
    probabilityValue = np.amax(predicciones)

    if probabilityValue > umbral:
        cv2.putText(imgOrignal, str(indiceClase) + " " + str(nombreClase(indiceClase)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
