from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np

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
    
    face_detector = YOLOv5('modelos/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('modelos/AntiSpoofing_bin_1.5_128.onnx')

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
        # predict score of Live face
        pred = make_prediction(frame, face_detector, anti_spoof)
        # if face is detected
        if pred is not None:
            (x1, y1, x2, y2), label, score = pred
            if label == 0:
                if score > 0.5:
                    res_text = "REAL      {:.2f}".format(score)
                    color = COLOR_REAL
                else: 
                    res_text = "unknown"
                    color = COLOR_UNKNOWN
            else:
                res_text = "FAKE      {:.2f}".format(score)
                color = COLOR_FAKE
                
            # Se dibuja la caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, rec_width)
            cv2.putText(frame, res_text, (x1, y1-txt_offset), 
                        cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, txt_width)            
            
        cv2.imshow('Face AntiSpoofing', frame)
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()