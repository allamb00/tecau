import cv2
import numpy as np
import os

def entrenar_modelo():
    # Cargar el clasificador de detección de caras pre-entrenado
    cascada_cara = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
        caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(caras) == 1:  # Asegurarse de que haya exactamente una cara en la imagen
            (x, y, w, h) = caras[0]
            rostros.append(gris[y:y+h, x:x+w])
            etiquetas.append(0)  # Asignar una etiqueta, por ejemplo, 0 para tu cara

    # Crear el modelo de reconocimiento facial
    modelo = cv2.face.LBPHFaceRecognizer_create()
    modelo.train(rostros, np.array(etiquetas))

    return modelo

def detectar_caras(modelo):
    # Cargar el clasificador de detección de caras pre-entrenado
    cascada_cara = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Inicializar la webcam
    camara = cv2.VideoCapture(0)
    
    # Crear la ventana
    cv2.namedWindow('Detección de caras')
    
    while True:
        # Leer el fotograma de la webcam
        ret, fotograma = camara.read()
        
        # Convertir el fotograma a escala de grises
        gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
        
        # Realizar la detección de caras en el fotograma
        caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))
        
        # Para cada cara detectada, intentar reconocerla
        for (x, y, w, h) in caras:
            # Obtener el rostro de la imagen en escala de grises
            rostro = gris[y:y+h, x:x+w]
            
            # Intentar reconocer el rostro utilizando el modelo entrenado
            etiqueta, porcentaje = modelo.predict(rostro)
            
            # Dibujar un rectángulo alrededor de la cara detectada
            cv2.rectangle(fotograma, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Mostrar el nombre correspondiente a la etiqueta reconocida
            if etiqueta == 0:  # Etiqueta correspondiente a tu cara
                nombre = "Andres"
            else:
                nombre = "Desconocido"
                
            texto_subtitulo = f"{nombre}: {porcentaje:.2f}%"

            # Dibujar el subtítulo en el fotograma con colores diferentes
            color_nombre = (0, 255, 0)  # Color verde para el nombre
            color_porcentaje = (200, 0, 255)  # Color rojo para el porcentaje
            cv2.putText(fotograma, nombre, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_nombre, 2)
            cv2.putText(fotograma, f"{porcentaje:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_porcentaje, 2)

        # Mostrar el fotograma con las caras detectadas y reconocidas
        cv2.imshow('Detección de caras', fotograma)
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la webcam y cerrar todas las ventanas
    camara.release()
    cv2.destroyAllWindows()

# Entrenar el modelo de reconocimiento facial
modelo = entrenar_modelo()

# Llamar a la función para iniciar la detección de caras y reconocimiento facial
detectar_caras(modelo)
