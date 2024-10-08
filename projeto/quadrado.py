import cv2
import time
import face_recognition
import threading

# Carregar o classificador de face pré-treinado do OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a webcam com resolução reduzida
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Reset to default exposure

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Inicializar variáveis de tempo
start_time = None
face_detected = False

# Inicializar o bot do Telegra

# Carregar a foto de referência e extrair as características do rosto
reference_image = face_recognition.load_image_file('reference.jpg')
reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

# Função para processar frames
def process_frame():
    global face_detected, start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converter o frame para escala de cinza (necessário para a detecção de faces)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Se um rosto for detectado, redefinir o temporizador
            face_detected = True
            start_time = time.time()  # Reiniciar o temporizador

            # Extrair a face do frame
            (x, y, w, h) = faces[0]
            face_frame = frame[y:y+h, x:x+w]
            
            # Detectar landmarks faciais
            face_landmarks_list = face_recognition.face_landmarks(face_frame)
            if face_landmarks_list:
                # Corrigir a localização da face para o formato (top, right, bottom, left)
                face_location = (y, x+w, y+h, x)
                face_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])
                if face_encodings:
                    match = face_recognition.compare_faces([reference_face_encoding], face_encodings[0])[0]
                    if match:
                        cv2.putText(frame, "Face matches reference", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Face does not match reference", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Desenhar um retângulo ao redor da face detectada
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            if face_detected and (time.time() - start_time) > 5:
               
                cv2.putText(frame, 'Volte ao trabalho!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                break

        # Mostrar o frame resultante
        cv2.imshow('Video', frame)

        # Sair do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Criar e iniciar a thread para processar frames
thread = threading.Thread(target=process_frame)
thread.start()

# Esperar a thread terminar
thread.join()

# Liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()