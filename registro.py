import cv2  # Biblioteca para captura e processamento de vídeo/imagens
import face_recognition  # Biblioteca para reconhecimento facial
import numpy as np  # Biblioteca para operações com arrays

def capture_reference_image(filename="reference.jpg"):
    # Inicializa a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # Captura um único quadro (frame) da webcam
    if ret:  # Verifica se a captura foi bem-sucedida
        cv2.imwrite(filename, frame)  # Salva o quadro capturado como uma imagem
    cap.release()  # Libera a webcam
    cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

def save_face_encoding(image_path="reference.jpg", encoding_file="face_encoding.npy"):
    # Carrega a imagem salva da face
    image = face_recognition.load_image_file(image_path)
    # Gera a codificação da face (um vetor de características únicas)
    face_encoding = face_recognition.face_encodings(image)[0]
    # Salva a codificação da face em um arquivo .npy para uso posterior
    with open(encoding_file, "wb") as f:
        np.save(f, face_encoding)

# Ponto de entrada do script
if __name__ == "__main__":
    capture_reference_image()  # Captura a imagem de referência da webcam
    save_face_encoding()  # Salva a codificação da face capturada
    print("Face registrada com sucesso!")  # Informa que a face foi registrada
