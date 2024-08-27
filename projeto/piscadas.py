import cv2  # Biblioteca para manipulação de imagens e vídeos
import dlib  # Biblioteca para detecção e manipulação de rostos
import imutils  # Biblioteca para manipulação de imagens e vídeos, como redimensionamento
import time  # Biblioteca para manipulação de tempo
from imutils import face_utils  # Funções auxiliares para manipulação de landmarks faciais
from scipy.spatial import distance as dist  # Biblioteca para calcular distâncias entre pontos
import face_recognition  # Biblioteca para reconhecimento facial

# Carregar a foto de referência e extrair as características do rosto
reference_image = face_recognition.load_image_file('reference.jpg')
reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

# Função para calcular a relação de aspecto do olho (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Distância vertical entre os pontos do olho
    B = dist.euclidean(eye[2], eye[4])  # Distância vertical entre os pontos do olho
    C = dist.euclidean(eye[0], eye[3])  # Distância horizontal entre os pontos do olho
    ear = (A + B) / (2.0 * C)  # Cálculo da relação de aspecto do olho (EAR)
    return ear

# Inicializar o detector de face do dlib e o preditor de landmarks faciais
detector = dlib.get_frontal_face_detector()  # Detector de rostos
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Predictor de landmarks faciais

# Defina as constantes do EAR
EYE_AR_THRESH = 0.25  # Limite para detectar uma piscada
EYE_AR_CONSEC_FRAMES = 0.1  # Número de frames consecutivos que deve estar abaixo do limiar para contar como piscada

# Contadores
COUNTER = 0  # Contador de frames onde o EAR está abaixo do limite
TOTAL = 0  # Contador total de piscadas
last_blink_time = None  # Tempo da última piscada
last_face_check_time = time.time()  # Tempo da última verificação de face

# Inicializar a webcam
cap = cv2.VideoCapture(0)  # Captura de vídeo da webcam

time_since_last_blink = None  # Tempo desde a última piscada
control = False  # Controle para verificar se a face corresponde à referência

while True:
    # Capturar frame a frame
    ret, frame = cap.read()  # Captura o frame da webcam
    frame = imutils.resize(frame, width=640)  # Redimensiona o frame para 640px de largura
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte o frame para escala de cinza

    # Detectar faces no frame
    faces = detector(gray, 0)  # Detecta rostos no frame em escala de cinza

    # Iterar sobre as faces detectadas
    for face in faces:
        shape = predictor(gray, face)  # Prever landmarks faciais
        shape = face_utils.shape_to_np(shape)  # Converter os landmarks para um array NumPy

        # Coordenadas dos olhos
        leftEye = shape[36:42]  # Coordenadas dos pontos do olho esquerdo
        rightEye = shape[42:48]  # Coordenadas dos pontos do olho direito

        # Calcular o EAR para ambos os olhos
        leftEAR = eye_aspect_ratio(leftEye)  # Relação de aspecto do olho esquerdo
        rightEAR = eye_aspect_ratio(rightEye)  # Relação de aspecto do olho direito

        # Média do EAR dos dois olhos
        ear = (leftEAR + rightEAR) / 2.0  # Média do EAR para os dois olhos

        # Desenhar o contorno dos olhos
        leftEyeHull = cv2.convexHull(leftEye)  # Desenha o contorno do olho esquerdo
        rightEyeHull = cv2.convexHull(rightEye)  # Desenha o contorno do olho direito
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # Exibe o contorno no frame
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # Exibe o contorno no frame

        # Verificar se é a mesma pessoa a cada 10 segundos
        if time.time() - last_face_check_time > 10:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte o frame para RGB
            face_encodings = face_recognition.face_encodings(rgb_frame)  # Codifica as características faciais

            if face_encodings:
                match = face_recognition.compare_faces([reference_face_encoding], face_encodings[0])[0]  # Compara a face atual com a referência
                if match:
                    cv2.putText(frame, "Face matches reference", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Exibe mensagem se a face corresponder
                else:
                    print("A face não corresponde à referência.")  # Mensagem se a face não corresponder
                    control = True  # Ativa o controle para sair do loop

            last_face_check_time = time.time()  # Atualiza o tempo da última verificação

        # Verificar se o EAR está abaixo do limite
        if ear < EYE_AR_THRESH:
            COUNTER += 1  # Incrementa o contador se o EAR estiver abaixo do limite
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1  # Incrementa o contador total de piscadas
                last_blink_time = time.time()  # Atualiza o tempo da última piscada
            COUNTER = 0  # Reseta o contador

        # Exibir o número total de piscadas
        cv2.putText(frame, f"Piscadas: {TOTAL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Exibe o total de piscadas no frame

        # Exibir o tempo da última piscada
        if last_blink_time is not None:
            time_since_last_blink = time.time() - last_blink_time  # Calcula o tempo desde a última piscada
            cv2.putText(frame, f"Ultima piscada: {time_since_last_blink:.2f} s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Exibe o tempo da última piscada no frame

    if time_since_last_blink is not None:
        if time_since_last_blink > 10:  # Verifica se o tempo desde a última piscada é maior que 10 segundos
            print("O funcionário nn piscou por mais de 10 segundos.")  # Mensagem de aviso
            break  # Sai do loop

    if control:
        break  # Sai do loop se a face não corresponder à referência

    # Mostrar o frame com os landmarks faciais e contagem de piscadas
    cv2.imshow("Frame", frame)  # Exibe o frame na tela

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Sai do loop se a tecla 'q' for pressionada

# Liberar a captura e fechar janelas
cap.release()  # Libera a captura de vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV
