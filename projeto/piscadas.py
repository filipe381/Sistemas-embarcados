import cv2
import dlib
import imutils
import time
import paho.mqtt.client as mqtt
from imutils import face_utils
from scipy.spatial import distance as dist
import face_recognition
import mediapipe as mp

# Configurações MQTT
broker = "localhost"
topic = "projeto/iot"

#Inicializando o mediapipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def load_reference_face(file_path):
    reference_image = face_recognition.load_image_file(file_path)
    reference_face_encoding = face_recognition.face_encodings(reference_image)[0]
    return reference_face_encoding

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_hand(frame):
    # Convertendo o frame para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return True
    return False

def face_detection_and_blinking():
    
    # Configurando o cliente MQTT
    client = mqtt.Client()
    client.connect(broker, 1883, 60)

    reference_face_encoding = load_reference_face('reference.jpg')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #Variáveis de sensibilidade
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 1

    COUNTER = 0
    TOTAL = 0
    last_blink_time = None
    last_face_check_time = time.time()
    last_face_detected_time = time.time()
    control = False
    cap = cv2.VideoCapture(0)

    #variáveis de desafio
    cooldownDesafio = 10
    ultimoDesafio = time.time()
    duracaoDesafio = None

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 0)

        if faces:
            last_face_detected_time = time.time()

            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[36:42]
                rightEye = shape[42:48]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if time.time() - last_face_check_time > 10:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_frame)

                    if face_encodings:
                        match = face_recognition.compare_faces([reference_face_encoding], face_encodings[0])[0]
                        if match:
                            cv2.putText(frame, "Face matches reference", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            client.publish(topic, "A face não corresponde à referência.")
                            control = True

                    last_face_check_time = time.time()

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        last_blink_time = time.time()
                    COUNTER = 0

                cv2.putText(frame, f"Piscadas: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if last_blink_time is not None:
                    time_since_last_blink = time.time() - last_blink_time
                    cv2.putText(frame, f"Ultima piscada: {time_since_last_blink:.2f} s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if time_since_last_blink > 10:
                        client.publish(topic, "O funcionário não piscou por mais de 10 segundos.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                #Desafios
                if last_blink_time is not None and time.time() - ultimoDesafio > cooldownDesafio:
                    ultimoDesafio = time.time()
                    duracaoDesafio = ultimoDesafio + 3 #para alterar a duração, altere o valor a direita
                    desafioFalhou = True 
                
                if duracaoDesafio is not None and time.time() < duracaoDesafio:
                    cv2.putText(frame, "Levante a mao", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if detect_hand(frame):
                        desafioFalhou = False #a pessoa levantou a mão e o desafio não falhou

                if duracaoDesafio is not None and time.time() >= duracaoDesafio:
                    if desafioFalhou:
                        client.publish(topic, "Desafio falhou: mão não levantada.")
                        cap.release()
                        cv2.destroyAllWindows()

        else:
            last_blink_time = time.time()  # Pausa o temporizador quando nenhum rosto é detectado

        if time.time() - last_face_detected_time > 20:
            client.publish(topic, "Nenhum rosto detectado por mais de 20 segundos.")
            break

        if control:
            break

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

face_detection_and_blinking()