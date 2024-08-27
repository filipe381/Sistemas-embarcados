import time
import cv2
import face_recognition
import dlib
from scipy.spatial import distance as dist
import numpy as np
from imutils import face_utils

def eye_aspect_ratio(eye):
    """Calcula a proporção da abertura dos olhos com base nos landmarks."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(frame, detector, predictor, eye_ar_thresh=0.25):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    blink_detected = False
    ear = None

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # Converte os landmarks em numpy array

        leftEye = shape[36:42]  # Índices dos landmarks do olho esquerdo
        rightEye = shape[42:48]  # Índices dos landmarks do olho direito
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        if ear < eye_ar_thresh:
            blink_detected = True
    
    return blink_detected, ear

def is_same_person(frame):
    tolerance = 0.6
    # Carrega a imagem salva
    reference_image = cv2.imread('reference.jpg')
    reference_encoding = face_recognition.face_encodings(reference_image)[0]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)
    if len(face_encodings) > 0:
        match = face_recognition.compare_faces([reference_encoding], face_encodings[0], tolerance=tolerance)
        return match[0]  # Retorna True se for a mesma pessoa, False caso contrário
    else:
        return False

def is_person_present(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return len(faces) > 0  # Retorna True se houver um rosto detectado, False caso contrário

def main():
    # Caminho da imagem salva
    saved_image_path = 'reference.jpg'

    # Inicializa o detector e o preditor de landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(0)

    last_blink_time = time.time()
    last_presence_time = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detecta se há uma pessoa presente
        if is_person_present(frame):
            last_presence_time = time.time()
            
            # Verifica se é a mesma pessoa da foto salva
            if not is_same_person(frame):
                print("Alerta: Não é a mesma pessoa!")
                break  # Encerra o programa
            
            # Detecta piscadas
            blink_detected, ear = detect_blink(frame, detector, predictor)
            if blink_detected:
                last_blink_time = time.time()
                print("Piscada detectada")
            
            # Verifica se a última piscada foi há mais de 10 segundos
            if time.time() - last_blink_time > 10:
                print("Alerta: Não detectou piscada nos últimos 10 segundos!")
                break  # Encerra o programa
        
        else:
            # Se a pessoa não for detectada por 20 segundos, envia uma mensagem de voz (mensagem simulada)
            if time.time() - last_presence_time > 20:
                print("Alerta: Pessoa não detectada por 20 segundos. Enviando mensagem de voz.")
                # Substitua este print por código para tocar uma mensagem de voz se necessário
                break  # Encerra o programa

        # Adicione um pequeno atraso para evitar sobrecarregar o sistema
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
