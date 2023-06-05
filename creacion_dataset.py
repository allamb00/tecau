'''
Autores: Raúl Baides, Andrés Llamosas y Amaia Echeandia
Asignatura: Tecnologías de Autenticación
Fecha: 06.06.2023

Descripción: Este código tiene como objetivo crear un conjunto de datos para un sistema de autenticación.
             La capturá se hará a través de la webcam habiendo indicado tanto el usuario como el número de capturas del vídeo.             

'''

import cv2
import os
from datetime import datetime

#Creamos una carpeta para almacenar los rostros si esta no existe
nombre_usuario = input("Introduce el nombre del usuario a registrar:  ")

if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print('Carpeta creada: ', 'dataset')

#Se crerá una carpeta nueva por cada usuario nuevo que se registre
if not os.path.exists(nombre_usuario):
    ruta = os.path.join('dataset', nombre_usuario)
    os.makedirs(ruta)
    print('Carpeta creada: ', nombre_usuario)
    
#Capturamos vídeo por la webcam
cap = cv2.VideoCapture(0)

#Cargamos el clasificador de rostros 'haarcascade'
faceClassif = cv2.CascadeClassifier('modelos/haarcascade_frontalface_default.xml')

numcapturas = int(input("Introduce el número de capturas a realizar:  "))

#Se capturarán imagenes del vídeo tantas veces se haya indicado y se detectarán los rostros
count = 0
while count < numcapturas:
    #Capturamos el frame 
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    #Transformamos cada imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Guardamos una copia para recortar los rostros más adelante
    auxFrame = frame.copy()
    
    #Detectamos las caras en escala de grises con el clasificador 
    faces = faceClassif.detectMultiScale(gray, 1.2, 9)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #Recortará y almacenará cada rostro identificado en la imagen
    for (x,y,w,h) in faces:
        #Recorta y redimensiona el rostro
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        
        #Capturamos fecha y hora para nombrar el archivo    
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        nombre_archivo = f"imagen_{count}{timestamp}.jpg"
        
        #Se almacena el rostro junto con el nombre del archivo
        cv2.imwrite(os.path.join(ruta, nombre_archivo), rostro)
        #Se va mostrando el rostro capturado
        cv2.imshow('rostro',rostro)
        count = count +1
        
        #Indicamos el número de captura actual del total de capturas introducido
        cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
        cv2.putText(frame,str(count)+'/'+str(numcapturas),(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)

#Liberamos recursos de captura y cerramos ventanas de visualización          
cap.release()
cv2.destroyAllWindows()