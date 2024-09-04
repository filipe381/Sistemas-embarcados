import cv2

def detect_reflection(image):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detecta áreas muito brilhantes
    _, bright_areas = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Conta o número de pixels brilhantes
    num_bright_pixels = cv2.countNonZero(bright_areas)
    
    # Define um limite para considerar a imagem como possivelmente sendo refletida em uma tela
    if num_bright_pixels > 500:
        print("Reflexos suspeitos detectados. Possível vídeo sendo exibido em uma tela.")
        return True
    else:
        print("Nenhum reflexo suspeito detectado.")
        return False

# Teste com um frame de vídeo
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    detect_reflection(frame)
cap.release()
cv2.destroyAllWindows()
