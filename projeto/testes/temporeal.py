import cv2
import numpy as np
import dlib
import json
import os
from datetime import datetime

# Função para calcular o brilho médio
def media(imagem):
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    brilho = np.mean(gray_image)
    return brilho

# Função para calcular o desvio padrão
def desvio_padrao(imagem):
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    desvio = np.std(gray_image)
    return desvio

# Função para detectar face
def detectar_face(imagem):
    detector = dlib.get_frontal_face_detector()
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    return len(faces) > 0

# Função para capturar imagem da webcam e mostrar média de brilho/desvio após 20 frames
def mostrar_brilho_webcam_tempo_real():
    # Acessar a webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    # Contadores para brilho, desvio padrão e detecção de face
    brilhos = []
    desvios = []
    detectado_count = 0
    total_frames = 0
    frames_salvos = []  # Lista para armazenar os caminhos dos frames salvos

    # Diretório para salvar os frames
    if not os.path.exists("frames"):
        os.makedirs("frames")

    while total_frames < 20:  # Continuar até capturar 20 frames
        # Capturar frame da webcam
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame da webcam.")
            break

        # Detectar rosto
        face_detectada = detectar_face(frame)
        total_frames += 1  # Contar o total de frames capturados

        # Calcular brilho e desvio padrão para cada frame, independentemente da detecção de face
        brilho = media(frame)
        desvio = desvio_padrao(frame)

        # Armazenar os valores nas listas
        brilhos.append(brilho)
        desvios.append(desvio)
        if face_detectada:
            detectado_count += 1  # Incrementar o contador de detecção

        # Criar um nome de arquivo único usando timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = f"frames/frame_{timestamp}.png"
        cv2.imwrite(frame_path, frame)
        frames_salvos.append(frame_path)  # Adicionar o caminho do frame à lista

        # Exibir o frame em uma janela
        cv2.imshow('Webcam - Detecção de Face', frame)

        # Pressionar 'q' para sair do loop manualmente
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calcular e exibir a média de brilho e desvio padrão
    media_brilho = np.mean(brilhos) if brilhos else 0
    media_desvio = np.mean(desvios) if desvios else 0

    # Calcular a porcentagem de detecção de face
    porcentagem_detectada = (detectado_count / total_frames) * 100 if total_frames > 0 else 0

    print("Resultados após capturar 20 frames:")
    print(f'Brilho médio: {media_brilho:.2f}')
    print(f'Desvio padrão médio: {media_desvio:.2f}')
    print(f'Total de frames capturados: {total_frames}')
    print(f'Porcentagem de detecção de face: {porcentagem_detectada:.2f}%')

    # Solicitar um ID para armazenar os dados
    interacao_id = input("Digite um ID para esta interação: ")

    # Criar um dicionário com os dados
    dados_interacao = {
        "id": interacao_id,
        "media_brilho": media_brilho,
        "media_desvio": media_desvio,
        "total_frames": total_frames,
        "porcentagem_detectada": porcentagem_detectada,
        "brilhos": brilhos,
        "desvios": desvios,
        "frames_salvos": frames_salvos
    }

    # Verificar se o arquivo de dados já existe
    if os.path.exists("dados_interacoes.json"):
        # Ler dados existentes
        with open("dados_interacoes.json", "r") as file:
            todas_interacoes = json.load(file)
    else:
        todas_interacoes = []

    # Adicionar os dados da interação
    todas_interacoes.append(dados_interacao)

    # Salvar os dados no arquivo JSON
    with open("dados_interacoes.json", "w") as file:
        json.dump(todas_interacoes, file, indent=4)

    print(f"Dados da interação salvos com o ID: {interacao_id}")

    # Liberar a webcam e fechar janelas
    cap.release()
    cv2.destroyAllWindows()

# Chamar a função
mostrar_brilho_webcam_tempo_real()
