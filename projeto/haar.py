import face_recognition
import cv2

# Carregar a imagem usando face_recognition
image = face_recognition.load_image_file('reference.jpg')

# Detectar os locais dos rostos na imagem (retorna uma lista de coordenadas de caixas delimitadoras)
face_locations = face_recognition.face_locations(image)

# Converter para grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Verificar se rostos foram detectados
if face_locations:
    for face_location in face_locations:
        # Obter as coordenadas do rosto detectado
        top, right, bottom, left = face_location

        # Selecionar a matriz correspondente ao rosto detectado na imagem em grayscale
        face_matrix = gray_image[top:bottom, left:right]

        # Exibir a matriz da seção do rosto
        print(f"Face detected in region: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")
        print(face_matrix)

        # Opcional: exibir o rosto detectado
        cv2.imshow('Detected Face', face_matrix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Nenhum rosto detectado na imagem.")
