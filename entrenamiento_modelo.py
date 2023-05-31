import cv2
import os
import numpy as np
from tqdm import tqdm

#Estructura del directorio:
# dataset
#   |-persona1
#   |    |-imagenes(.jpg)
#   |-persona2
#        |-imagenes(.jpg)

dataPath = 'dataset' #Ruta al dataset
peopleList = os.listdir(dataPath)

# Filtrar solo los directorios de la lista
peopleList = [item for item in peopleList if os.path.isdir(os.path.join(dataPath, item))]

print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    lista_caras_registradas = [archivo for archivo in os.listdir(personPath) if archivo.endswith('.jpg')]

    with tqdm(total=len(lista_caras_registradas), desc="Progreso de lectura y clasificación de imágenes") as pbar:
        for fileName in lista_caras_registradas:
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            pbar.update(1)  # Actualizar el progreso de la barra
        label = label + 1
        
# Se selecciona el método para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Se entrena el modelo reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

if not os.path.exists('modelos'):
    os.makedirs('modelos')
    print('Carpeta creada: ', 'modelos')

# Se almacena el modelo obtenido
#face_recognizer.write('modelos/modeloEigenFace.xml')
#face_recognizer.write('modelos/modeloFisherFace.xml')
face_recognizer.write('modelos/modeloLBPHFace.xml')
print("Modelo almacenado")