import cv2
import numpy as np
import dlib

# Função para calcular o brilho médio
def media(imagem):
    # Converter a imagem para grayscale
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Calcular o brilho como a média dos valores de intensidade dos pixels
    brilho = np.mean(gray_image)
    return brilho

# Função para calcular o desvio padrão
def desvio_padrao(imagem):
    # Converter a imagem para grayscale
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Calcular o desvio padrão dos valores de intensidade dos pixels
    desvio = np.std(gray_image)
    return desvio

def detectar_face(imagem):
    # Inicializar o detector de faces do dlib
    detector = dlib.get_frontal_face_detector()
    
    # Converter a imagem para grayscale
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces
    faces = detector(gray_image)

    # Verificar se alguma face foi detectada
    if len(faces) > 0:
        return True
    else:
        return False

# Função para capturar a imagem da webcam, detectar o rosto, e calcular brilho/desvio
def mostrar_brilho_webcam():
    # Acessar a webcam (0 corresponde à câmera padrão do sistema)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    # Ler um frame da webcam
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem da webcam.")
        cap.release()
        return

    # Liberar a câmera após capturar o frame
    cap.release()

    # Calcular o brilho
    brilho = media(frame)
    desvio = desvio_padrao(frame)

    # Exibir os valores no terminal
    print("face detectada" if detectar_face(frame) else "face não detectada")
    print(f'Brilho da imagem: {brilho:.2f}')
    print(f'Desvio padrão do brilho: {desvio:.2f}')

# Usar a função
mostrar_brilho_webcam()
