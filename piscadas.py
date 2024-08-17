import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist

# Função para calcular a relação de aspecto do olho (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Inicializar o detector de face do dlib e o preditor de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Defina as constantes do EAR
EYE_AR_THRESH = 0.25  # Limite para detectar uma piscada
EYE_AR_CONSEC_FRAMES = 3  # Número de frames consecutivos que deve estar abaixo do limiar para contar como piscada

# Contadores
COUNTER = 0
TOTAL = 0

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame a frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    faces = detector(gray, 0)

    # Iterar sobre as faces detectadas
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Coordenadas dos olhos
        leftEye = shape[36:42]
        rightEye = shape[42:48]
        
        # Calcular o EAR para ambos os olhos
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Média do EAR dos dois olhos
        ear = (leftEAR + rightEAR) / 2.0
        
        # Desenhar o contorno dos olhos
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Verificar se o EAR está abaixo do limite
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0
        
        # Exibir o número total de piscadas
        cv2.putText(frame, f"Piscadas: {TOTAL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Mostrar o frame com os landmarks faciais e contagem de piscadas
    cv2.imshow("Frame", frame)
    
    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()
