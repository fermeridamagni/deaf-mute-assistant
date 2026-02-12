""""Test de reconocimiento de gestos con MediaPipe en Mac M4.
- Usa la cámara de tu MacBook para detectar la mano.
- Muestra en pantalla el conteo de dedos y el estado (Encender/Apagar).
- Imprime en consola cuando se envía un comando (simulado).
- Para salir, presiona 'q' en la ventana de video.
NOTA: Asegúrate de tener el modelo "hand_landmarker.task" en la misma carpeta que este script.
"""

import os
import time
import cv2
import mediapipe as mp

# --- CONFIGURACIÓN PARA MACBOOK ---
# En macOS, el índice 0 suele ser la cámara FaceTime HD.
# NOTA: Si tienes un iPhone cerca y usas "Cámara de Continuidad",
# a veces el iPhone toma el índice 0 o 1. Si no abre la de la Mac, prueba cambiar a 1.
CAPTURA_INDEX = 0

# --- Nueva API de MediaPipe Tasks ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Conexiones de la mano para dibujar el esqueleto
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # Anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
]


def draw_hand_landmarks(frame, landmarks, connections):
    """Dibuja los landmarks y conexiones de la mano sobre el frame."""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in connections:
        # Dibujar línea entre los puntos conectados
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        # Dibujar círculo en cada landmark
        cv2.circle(frame, point, 5, (0, 0, 255), -1)


def detectar_dedos_levantados(landmarks):
    """"Detecta cuántos dedos están levantados basándose en la posición de los landmarks.
- Compara la posición de las puntas (Tips) con las bases (PIPs) para cada dedo.
- Para el pulgar, se compara la distancia horizontal entre la punta y la base.
"""

    dedos = []

    # IDs de las puntas (Tips) y bases (PIPs)
    # 8: Índice, 12: Medio, 16: Anular, 20: Meñique

    # Índice
    if landmarks[8].y < landmarks[6].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Medio
    if landmarks[12].y < landmarks[10].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Anular
    if landmarks[16].y < landmarks[14].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Meñique
    if landmarks[20].y < landmarks[18].y:
        dedos.append(1)
    else:
        dedos.append(0)

    # Pulgar (Lógica simplificada para modo espejo)
    if abs(landmarks[4].x - landmarks[20].x) > 0.15:
         dedos.append(1)
    else:
         dedos.append(0)

    return sum(dedos)


# Configuración del Hand Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAPTURA_INDEX)

# Definir tamaño de ventana (opcional, para que se vea bien en pantallas Retina)
cap.set(3, 1280)
cap.set(4, 720)

estado_actual = ""

try:
    print("Iniciando cámara...")
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo recibir frame. ¿Cámara ocupada?")
                break

            # 1. ESPEJO: Invertir horizontalmente para que sea natural (como un espejo)
            frame = cv2.flip(frame, 1)

            # 2. Convertir BGR a RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. Crear imagen MediaPipe y procesar
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                for landmarks in results.hand_landmarks:
                    draw_hand_landmarks(frame, landmarks, HAND_CONNECTIONS)

                    total_dedos = detectar_dedos_levantados(landmarks)

                    # --- LÓGICA DE CONTROL ---
                    if total_dedos >= 4: # Mano Abierta
                        mensaje = "ENCENDER LUZ"
                        color = (0, 255, 0) # Verde

                        if estado_actual != "ENCENDER":
                            print(f"--> COMANDO ENVIADO: {mensaje}")
                            estado_actual = "ENCENDER"

                    elif total_dedos <= 1: # Puño Cerrado
                        mensaje = "APAGAR LUZ"
                        color = (0, 0, 255) # Rojo

                        if estado_actual != "APAGAR":
                            print(f"--> COMANDO ENVIADO: {mensaje}")
                            estado_actual = "APAGAR"

                    else:
                        mensaje = "ESPERANDO..."
                        color = (200, 200, 200)

                    # Poner texto en pantalla
                    cv2.putText(frame, f"Estado: {mensaje}", (30, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                    # Mostrar conteo de dedos (debug)
                    cv2.putText(frame, f"Dedos: {total_dedos}", (30, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Prueba Mac M4 - Control por Señas', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
