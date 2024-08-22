import cv2
import time
import telegram
import idTelegram
import asyncio

# Carregar o classificador de face pré-treinado do OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a webcam
cap = cv2.VideoCapture(0)

# Inicializar variáveis de tempo
start_time = None
face_detected = False

# Inicializar o bot do Telegram
bot_token = '7390853298:AAFEZhfjtMaB7lfNGk4NjjWsQxNYp42olUs'
chat_id = asyncio.run(idTelegram.get_chat_id())
bot = telegram.Bot(token=bot_token)

while True:
    # Capturar frame a frame
    ret, frame = cap.read()

    # Converter o frame para escala de cinza (necessário para a detecção de faces)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Se um rosto for detectado, redefinir o temporizador
        face_detected = True
        start_time = time.time()  # Reiniciar o temporizador
    else:
        # Se nenhum rosto for detectado
        if face_detected:
            face_detected = False
            start_time = time.time()  # Iniciar o temporizador

        if start_time is not None:  # Verifique se start_time foi inicializado
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5:
                # Exibir a mensagem na tela
                cv2.putText(frame, 'Volte ao trabalho!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Enviar mensagem pelo Telegram
                asyncio.run(bot.send_message(chat_id=chat_id, text='Nenhum rosto detectado!'))

    # Desenhar um retângulo ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar o frame resultante
    cv2.imshow('Video', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()
