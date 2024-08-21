predictor_path = "d:\\git\\Sistemas embarcados\\shape_predictor_68_face_landmarks.dat"
import cv2
import dlib
import imutils
from imutils import face_utils

# Carregar o detector de faces do dlib e o preditor de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame a frame
    ret, frame = cap.read()
    
    # Redimensionar o frame para processamento mais rápido
    frame = imutils.resize(frame, width=640)
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces no frame
    faces = detector(gray, 0)
    
    # Iterar sobre as faces detectadas
    for face in faces:
        # Prever landmarks faciais
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Desenhar círculos nos landmarks faciais
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Mostrar o frame com os landmarks faciais
    cv2.imshow("Frame", frame)
    
    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()
