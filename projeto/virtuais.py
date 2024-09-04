import psutil

def check_emulator_processes():
    # Lista de processos conhecidos de emuladores
    emulators = ["ManyCam", "obs64", "xsplit.core", "webcamoid"]

    # Obtém a lista de processos em execução
    for process in psutil.process_iter(['name']):
        for emulator in emulators:
            if emulator.lower() in process.info['name'].lower():
                print(f"Emulador de webcam detectado: {process.info['name']}")
                return True

    print("Nenhum emulador de webcam detectado.")
    return False

if __name__ == "__main__":
    if check_emulator_processes():
        print("Encerrando o programa devido à detecção de emulador.")
    else:
        print("Continuando a execução normal.")
check_emulator_processes()