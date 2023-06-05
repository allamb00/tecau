'''
Autores: Raúl Baides, Andrés Llamosas y Amaia Echeandia
Asignatura: Tecnologías de Autenticación
Fecha: 06.06.2023

Descripción: Este código tiene como objetivo entrenar el conjunto de datos creado por 'creacion_dataset.py'
             con el método LBPH (Local Binary Patterns Histograms) asignando etiquetas y reconociendo rostros.

'''

import cv2
import os
import numpy as np
from tqdm import tqdm

'''
#Estructura del directorio:
# dataset
#   |-persona1
#   |    |-imagenes(.jpg)
#   |-persona2
#        |-imagenes(.jpg)
'''

dataPath = 'dataset' #Ruta al dataset
peopleList = os.listdir(dataPath)

# Filtrar solo los directorios de la lista
peopleList = [item for item in peopleList if os.path.isdir(os.path.join(dataPath, item))]

#Lista los usuarios registrados
print('Lista de personas: ', peopleList)

#Declaración de arrays para etiquetado y reconocimiento
labels = []
facesData = []
label = 0


for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    
    #Filtrar solo por los archivos 'jpg' del directorio de cada usuario
    lista_caras_registradas = [archivo for archivo in os.listdir(personPath) if archivo.endswith('.jpg')]

    with tqdm(total=len(lista_caras_registradas), desc="Progreso de lectura y clasificación de imágenes") as pbar:
        for fileName in lista_caras_registradas:
        
            #Asignación de una etiqueta a los rostros de cada persona
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            
            pbar.update(1)  #Actualizar el progreso de la barra
            
        label = label + 1
        
#Se selecciona el método LBPH para entrenar el reconocedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Se entrena el modelo reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

#Se crea la carpeta con los modelos si no existiera
if not os.path.exists('modelos'):
    os.makedirs('modelos')
    print('Carpeta creada: ', 'modelos')

#Se almacena el modelo obtenido en formato xml
face_recognizer.write('modelos/modeloLBPHFace.xml')
print("Modelo almacenado")