import cv2
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def entrenar_modelo():
    # Inicializar la lista de rostros y etiquetas
    rostros = []
    etiquetas = []

    # Leer las imágenes de entrenamiento
    ruta_dataset = 'dataset/'
    lista_imagenes = [archivo for archivo in os.listdir(ruta_dataset) if archivo.endswith('.jpg')]

    for nombre_imagen in lista_imagenes:
        print(nombre_imagen)
        ruta_imagen = os.path.join(ruta_dataset, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Ajustar el tamaño de la imagen a un tamaño específico
        nuevo_tamano = (100, 100)  # Tamaño deseado
        gris_redimensionado = cv2.resize(gris, nuevo_tamano)

        rostros.append(gris_redimensionado.flatten())  # Aplanar la imagen de la cara como vector de características
        
        etiquetas.append(0)  # Asignar una etiqueta, por ejemplo, 0 para tu cara

    # Crear el clasificador débil utilizando Decision Tree Classifier
    clasificador_debil = DecisionTreeClassifier(max_depth=1)

    # Crear el clasificador AdaBoost utilizando el clasificador débil
    clasificador_adaboost = AdaBoostClassifier(base_estimator=clasificador_debil, n_estimators=100)

    # Entrenar el clasificador AdaBoost con las imágenes y etiquetas
    clasificador_adaboost.fit(rostros, etiquetas)

    return clasificador_adaboost

def detectar_caras(modelo):
    # Inicializar la webcam
    camara = cv2.VideoCapture(0)

    # Crear la ventana
    cv2.namedWindow('Detección de caras')

    while True:
        # Leer el fotograma de la webcam
        ret, fotograma = camara.read()

        # Convertir el fotograma a escala de grises
        gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

        # Ajustar el tamaño de la imagen a un tamaño específico
        nuevo_tamano = (100, 100)  # Tamaño deseado
        gris_redimensionado = cv2.resize(gris, nuevo_tamano)
        
        # Preprocesar la imagen de entrada si es necesario

        # Obtener las regiones candidatas a caras utilizando AdaBoost
        caras_detectadas = modelo.predict(gris_redimensionado.flatten().reshape(1, -1))
        caras_detectadas = caras_detectadas.reshape((gris_redimensionado.shape[0], gris_redimensionado.shape[1]))

        # Para cada cara detectada, dibujar un rectángulo verde
        for (x, y, w, h) in cv2.boundingRect(caras_detectadas):
            cv2.rectangle(fotograma, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Mostrar el fotograma con las caras detectadas y reconocidas
        cv2.imshow('Detección de caras', fotograma)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la webcam y cerrar todas las ventanas
    camara.release()
    cv2.destroyAllWindows()

# Entrenar el modelo de reconocimiento facial utilizando AdaBoost
modelo = entrenar_modelo()

# Llamar a la función para iniciar la detección de caras y reconocimiento facial
detectar_caras(modelo)
