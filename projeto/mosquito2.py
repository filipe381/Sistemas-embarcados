import paho.mqtt.client as mqtt
import time

# Definindo o tópico e o host
broker = "test.mosquitto.org"
topic = "umidade/sensor"
connected = False

# Função chamada quando a conexão ao broker é estabelecida
def on_connect(client, userdata, flags, rc):
    global connected
    if rc == 0:
        print(f"Conectado ao broker com sucesso, código de resultado {rc}")
        client.subscribe(topic)
        connected = True
    else:
        print(f"Falha na conexão. Código de resultado: {rc}")
        connected = False

# Função chamada quando uma mensagem é recebida no tópico
def on_message(client, userdata, msg):
    print(f"Mensagem recebida: {msg.payload.decode()} no tópico: {msg.topic}")

# Criando o cliente MQTT
client = mqtt.Client()

# Configurando callbacks
client.on_connect = on_connect
client.on_message = on_message

# Loop de tentativa de conexão
while not connected:
    try:
        print("Tentando se conectar ao broker...")
        client.connect(broker, 1883, 60)  # Conecta ao broker
        client.loop_start()  # Inicia o loop em background
        while not connected:  # Verifica se está conectado
            print("Aguardando conexão...")
            time.sleep(1)  # Aguarda um segundo antes de verificar novamente
    except Exception as e:
        print(f"Erro ao tentar conectar: {e}")
        time.sleep(5)  # Aguarda 5 segundos antes de tentar reconectar

# Mantendo o loop para continuar ouvindo o tópico
try:
    while True:
        time.sleep(1)  # Mantém o programa ativo para continuar ouvindo
except KeyboardInterrupt:
    print("Desconectando...")
    client.loop_stop()  # Para o loop de eventos MQTT
    client.disconnect()  # Desconecta do broker
