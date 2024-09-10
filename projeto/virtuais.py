import psutil

def check_emulator_processes():
    # Lista de processos conhecidos de emuladores e webcams virtuais
    emulators = [
        "ManyCam", "obs64", "xsplit.core", "webcamoid", "YouCam", "CamTwist",
        "SplitCam", "Snap Camera", "DroidCam", "EpocCam", "ManyCamService",
        "VirtualCam", "v4l2loopback"
    ]

    # Obtém a lista de processos em execução
    for process in psutil.process_iter(['name']):
        for emulator in emulators:
            if emulator.lower() in process.info['name'].lower():
                return True

    return False
