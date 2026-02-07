import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- CONFIGURATIE ---
MODEL_PATH = 'hand_landmarker.task'

def verwerk_resultaat(result):
    """
    Hulpfunctie om gebaren te herkennen uit de landmarks.
    In de Tasks API is de data gestructureerd als result.hand_landmarks.
    """
    if not result.hand_landmarks:
        return "Geen hand"
    
    # Pak de eerste gedetecteerde hand
    landmarks = result.hand_landmarks[0]
    
    # Voorbeeld logica: Wijsvinger (punt 8) hoger dan gewricht (punt 6)
    if landmarks[8].y < landmarks[6].y:
        return "Wijsvinger Omhoog"
    return "Vuist / Anders"

# --- INITIALISATIE ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, # Nodig voor webcam stream
    num_hands=1
)

# Gebruik 'with' zodat de detector netjes wordt afgesloten
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # MediaPipe Tasks verwacht een mp.Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Voor de VIDEO modus hebben we een timestamp nodig in milliseconden
        timestamp_ms = int(time.time() * 1000)
        
        # Voer de detectie uit
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Gebaar herkennen en actie koppelen
        actie = verwerk_resultaat(detection_result)
        
        # Visuele feedback
        cv2.putText(frame, f"Gebaar: {actie}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Tasks - Hand Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()