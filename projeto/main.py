import numpy as np
import cv2

# Carregar a imagem
img = cv2.imread('reference.jpg')

# Converter a imagem para grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ajustar as opções de impressão do NumPy para exibir a matriz completa
np.savetxt('gray_image_matrix.txt', gray_img, fmt='%d')

# Exibir a matriz completa da imagem grayscale
print(gray_img)
