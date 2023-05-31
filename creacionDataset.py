import cv2
import os
from datetime import datetime

#Creamos una carpeta para almacenar los rostros si esta no existe
nombre_usuario = input("Introduce el nombre del usuario a registrar:  ")

if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print('Carpeta creada: ', 'dataset')
    
if not os.path.exists(nombre_usuario):
    ruta = os.path.join('dataset', nombre_usuario)
    os.makedirs(ruta)
    print('Carpeta creada: ', nombre_usuario)
    
    
cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier('modelos/haarcascade_frontalface_default.xml')

numcapturas = int(input("Introduce el número de capturas a realizar:  "))

count = 0
while count < numcapturas:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    
    faces = faceClassif.detectMultiScale(gray, 1.2, 9)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
                            
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        nombre_archivo = f"imagen_{count}{timestamp}.jpg"
        
        #if k == ord('s'):
        cv2.imwrite(os.path.join(ruta, nombre_archivo), rostro)
        cv2.imshow('rostro',rostro)
        count = count +1
        #5
        cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
        cv2.putText(frame,str(count)+'/'+str(numcapturas),(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
            
cap.release()
cv2.destroyAllWindows()