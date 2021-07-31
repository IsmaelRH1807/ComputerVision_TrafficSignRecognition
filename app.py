from __future__ import division, print_function
# coding=utf-8

import os
import cv2
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Definimos una app de FLask
app = Flask(__name__)

# Ruta del modelo
MODEL_PATH = 'model_trained.h5'

# Cargamos el modelo
model = load_model(MODEL_PATH)

print('Servidor listo: http://127.0.0.1:5000/')

# Funciones de preprocesado de la imagen
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

# Función previa a la evaluación con el modelo
def preparar_imagen(img_path):

    #Cargamos la imagen guardada
    img = image.load_img(img_path, target_size=(1280, 720))

    #Preprocesamos la imagen antes de evaluarla con el modelo
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    return img

#Devuelve el nombre de la clase detectada
def nombreClase(classNo):
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

# Con el método GET carga la página
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

# Con la ruta '/predict'
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Saco el archivo mediante POST
        f = request.files['file']

        # Guardo el archivo en la ruta 'uploads'
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Hago la predicción con el modelo
        img =  preparar_imagen(file_path)
        predicciones = model.predict(img)
        indiceClase = np.argmax(predicciones)

        # String con el resultado
        result = str(indiceClase) + " " + str(nombreClase(indiceClase))

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

