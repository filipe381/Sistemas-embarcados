import paho.mqtt.client as mqtt

# Definindo o tópico e o host
broker = "172.20.10.3"
topic = "projeto/embarcados"

# Função chamada quando a conexão ao broker é estabelecidas
def on_connect(client, userdata, flags, rc):
    print(f"Conectado ao broker com código de resultado {rc}")
    # Se inscrever no tópico
    client.subscribe(topic)
    print(f"Inscrito no tópico: {topic}")

# Função chamada quando uma mensagem é recebida no tópico
def on_message(client, userdata, msg):
    print(f"Mensagem recebida: {msg.payload.decode()} no tópico: {msg.topic}")

# Criando o cliente MQTT
client = mqtt.Client()

# Configurando callbacks
client.on_connect = on_connect
client.on_message = on_message

# Conectando ao broker
client.connect(broker, 1883, 60)

# Mantendo o loop para continuar ouvindo o tópico
client.loop_forever()