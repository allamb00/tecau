from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np
import os

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 
    bbox = face_detector([img])[0]
    
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    
    return bbox, label, score

if __name__ == "__main__":    

    dataPath = 'dataset' #Ruta al dataset
    peopleList = os.listdir(dataPath)

    # Filtrar solo los directorios de la lista
    peopleList = [item for item in peopleList if os.path.isdir(os.path.join(dataPath, item))]

    print('Lista de personas: ', peopleList)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Leyendo los modelos
    face_recognizer.read('modelos/modeloLBPHFace.xml')
    face_detector = YOLOv5('modelos/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('modelos/AntiSpoofing_bin_1.5_128.onnx')
    faceClassif = cv2.CascadeClassifier('modelos/haarcascade_frontalface_default.xml') 

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    print('Frame size  :', frame_size)
    
    # process frames
    rec_width = max(1, int(frame_width/240))
    txt_offset = int(frame_height/50)
    txt_width = max(1, int(frame_width/480))   

    

    while True:
        ret,frame = cap.read()
        if ret == False: break 
    
        # DETECCIÓN DE SPOOFING
        pred = make_prediction(frame, face_detector, anti_spoof) 
        
        liveness = pred
        # Si existe una cara
        if pred is not None:
            (x1, y1, x2, y2), _ , liveness = pred
            
        #RECONOCIMIENTO DE ROSTRO    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        faces = faceClassif.detectMultiScale(gray,1.3,5)
    
        for (x,y,w,h) in faces:
            rostro = gray[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            reconocimiento = face_recognizer.predict(rostro)
    
            cv2.putText(frame,'{:.10f}'.format(reconocimiento[1]),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.putText(frame,'{:.10f}'.format(liveness*100),(x,y-50),1,1.3,(255,255,0),1,cv2.LINE_AA)
            
            # Se reconoce una cara real
            if reconocimiento[1] < 70 and liveness > 0.5:
                cv2.putText(frame,'{}'.format(peopleList[reconocimiento[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.putText(frame,'Real',(x,y-70),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                
            # Se reconoce una cara falsa   
            elif reconocimiento[1] < 70 and liveness <= 0.5:
                cv2.putText(frame,'{}'.format(peopleList[reconocimiento[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.putText(frame,'Falsa',(x,y-70),2,1.1,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            # No se reconoce una cara real   
            elif reconocimiento[1] >= 70 and liveness > 0.5:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.putText(frame,'Real',(x,y-70),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            # No se reconoce una cara falsa    
            elif reconocimiento[1] >= 70 and liveness < 0.5:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.putText(frame,'Falsa',(x,y-70),2,1.1,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                        
        cv2.imshow('Face AntiSpoofing', frame)
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()