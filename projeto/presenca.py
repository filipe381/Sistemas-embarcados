import cv2
import time

def monitor_face_presence():
    # Carregar o classificador de face pré-treinado do OpenCV (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializar a webcam
    cap = cv2.VideoCapture(0)

    # Inicializar variáveis de tempo
    start_time = None
    face_detected = False

    while True:
        # Capturar frame a frame
        ret, frame = cap.read()

        # Converter o frame para escala de cinza (necessário para a detecção de faces)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Se um rosto for detectado, redefinir o temporizador
            face_detected = True
            start_time = time.time()  # Reiniciar o temporizador
        else:
            # Se nenhum rosto for detectado
            if face_detected:
                face_detected = False
                start_time = time.time()  # Iniciar o temporizador

            if start_time is not None:  # Verifique se start_time foi inicializado
                elapsed_time = time.time() - start_time
                if elapsed_time >= 10:
                   return False
                
                
monitor_face_presence()