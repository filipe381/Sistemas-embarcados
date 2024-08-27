import cv2  # Biblioteca para captura e processamento de vídeo/imagens
import face_recognition  # Biblioteca para reconhecimento facial
import numpy as np  # Biblioteca para operações com arrays

def load_face_encoding(encoding_file="face_encoding.npy"):
    # Carrega a codificação da face previamente salva a partir do arquivo .npy
    with open(encoding_file, "rb") as f:
        face_encoding = np.load(f)
    return face_encoding  # Retorna a codificação da face

def compare_face(reference_encoding, tolerance=0.6):
    # Inicializa a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # Captura um único quadro (frame) da webcam
    cap.release()  # Libera a webcam

    if ret:  # Verifica se a captura foi bem-sucedida
        # Converte a imagem de BGR (padrão OpenCV) para RGB (padrão face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Gera as codificações faciais para todas as faces detectadas no quadro
        face_encodings = face_recognition.face_encodings(rgb_frame)

        # Verifica se alguma face foi detectada
        if len(face_encodings) > 0:
            # Compara a face detectada com a face de referência
            match = face_recognition.compare_faces([reference_encoding], face_encodings[0], tolerance=tolerance)
            if match[0]:  # Se houver correspondência, é a mesma pessoa
                print("A mesma pessoa foi detectada!")
            else:  # Caso contrário, é uma pessoa diferente
                print("Outra pessoa foi detectada!")
        else:
            print("Nenhuma face detectada!")  # Se nenhuma face for detectada
    else:
        print("Erro ao capturar a imagem.")  # Se houver falha na captura

# Ponto de entrada do script
if __name__ == "__main__":
    reference_encoding = load_face_encoding()  # Carrega a codificação da face de referência
    compare_face(reference_encoding)  # Compara a face capturada com a face de referência