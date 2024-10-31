import cv2
import numpy as np

def calcular_contraste(imagem):
    # Converter a imagem para grayscale
    gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Calcular o valor máximo e mínimo da imagem
    valor_maximo = np.max(gray_image)
    valor_minimo = np.min(gray_image)

    # Calcular o contraste como a diferença entre o máximo e o mínimo
    contraste = valor_maximo - valor_minimo

    return contraste

def mostrar_contraste(imagem_path):
    # Carregar a imagem
    imagem = cv2.imread(imagem_path)

    # Calcular o contraste
    contraste = calcular_contraste(imagem)

    # Mostrar a imagem
    cv2.imshow('Imagem', imagem)

    # Exibir o valor do contraste no terminal
    print(f'Contraste da imagem: {contraste}')

    # Esperar por uma tecla e fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usar a função
mostrar_contraste('reference.jpg')
