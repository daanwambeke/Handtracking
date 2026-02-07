"""
Hand Tracking Muisbesturing met Wijsvinger
- Gebruikt MediaPipe Hand Landmarker
- Wijsvinger beweegt de muis
- Duim+wijsvinger samenknijpen = klik
- Alle vingers omhoog = reset/kalibratie
- Rechtermuisknop met ringvinger+pink omhoog
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import core
from mediapipe.tasks.python.vision import hand_landmarker, RunningMode
import pyautogui
import numpy as np
import time

# =======================
# CONFIGURATIE
# =======================
MODEL_PATH = "hand_landmarker.task"
MAX_HANDS = 1  # 1 hand voor muisbesturing

# PyAutoGUI veiligheid
pyautogui.FAILSAFE = False  # Uitgeschakeld - muis stopt niet in hoeken
pyautogui.PAUSE = 0.001

# Schermafmetingen
screen_w, screen_h = pyautogui.size()

# Smoothing parameters
SMOOTHING = 0.4  # Lager = sneller en directer (was 0.7)
SENSITIVITY = 1.3  # Vermenigvuldiger voor snelheid (>1 = sneller)
prev_x, prev_y = 0, 0

# Camera frame parameters
FRAME_REDUCTION = 2  # Pixels van rand negeren (kleiner = groter actief gebied!)
cam_w, cam_h = 640, 480

# Performance tracking
fps_start_time = time.time()
fps_counter = 0
current_fps = 0

# =======================
# HAND LANDMARKER SETUP
# =======================
options = hand_landmarker.HandLandmarkerOptions(
    base_options=core.base_options.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = hand_landmarker.HandLandmarker.create_from_options(options)

# =======================
# WEBCAM SETUP
# =======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
timestamp = 0

# =======================
# HELPER FUNCTIES
# =======================

# Landmark indexen
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

WRIST = 0

def get_distance(point1, point2):
    """Euclidische afstand tussen twee landmarks"""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_finger_up(tip, pip, wrist, hand):
    """Check of een vinger omhoog is"""
    return hand[tip].y < hand[pip].y

def count_fingers(hand):
    """Tel aantal vingers omhoog"""
    count = 0
    fingers = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    
    for tip, pip in zip(fingers, pips):
        if is_finger_up(tip, pip, WRIST, hand):
            count += 1
    
    # Duim apart checken (horizontaal)
    if hand[THUMB_TIP].x < hand[THUMB_IP].x:  # Voor rechterhand
        count += 1
    
    return count

def detect_pinch(hand):
    """Detecteer duim+wijsvinger knijpgebaar"""
    distance = get_distance(hand[THUMB_TIP], hand[INDEX_TIP])
    return distance < 0.06  # Iets ruimer voor betere detectie (was 0.05)

def detect_gesture_mode(hand):
    """
    Detecteer besturingsmodus:
    - Index omhoog alleen = Muis bewegen
    - Duim+Index knijpen = Linker klik
    - Index + Middel omhoog = Scrollen
    - Ringvinger + Pink omhoog (ZONDER index/middel) = Rechter klik
    - Alle vingers omhoog = Pauze/Reset
    """
    # Check pinch EERST - dit is de belangrijkste check
    pinch = detect_pinch(hand)
    if pinch:
        return "CLICK"
    
    # Tel vingers
    fingers_up = count_fingers(hand)
    index_up = is_finger_up(INDEX_TIP, INDEX_PIP, WRIST, hand)
    middle_up = is_finger_up(MIDDLE_TIP, MIDDLE_PIP, WRIST, hand)
    ring_up = is_finger_up(RING_TIP, RING_PIP, WRIST, hand)
    pinky_up = is_finger_up(PINKY_TIP, PINKY_PIP, WRIST, hand)
    
    # Check andere gebaren
    if fingers_up >= 5:
        return "PAUSE"
    elif ring_up and pinky_up and not index_up and not middle_up:
        return "RIGHT_CLICK"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "SCROLL"
    elif index_up and not middle_up and not ring_up:
        return "MOVE"
    else:
        return "IDLE"

# =======================
# MAIN LOOP
# =======================
click_cooldown = 0
right_click_cooldown = 0
mode = "MOVE"
scroll_start_y = None  # Voor scroll tracking
double_click_timer = 0
last_click_time = 0

print("Hand Muisbesturing Gestart!")
print("Instructies:")
print("   - Wijsvinger omhoog = Muis bewegen")
print("   - Duim + wijsvinger samenknijpen = Linker klik")
print("   - Wijsvinger + middelvinger omhoog = Scrollen")
print("   - Ringvinger + pink omhoog = Rechter klik")
print("   - Alle vingers omhoog = Pauzeer besturing")
print("   - ESC of X-knop = Stoppen")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Spiegel het beeld voor natuurlijker gevoel
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Teken actieve zone
    cv2.rectangle(frame, (FRAME_REDUCTION, FRAME_REDUCTION), 
                  (w - FRAME_REDUCTION, h - FRAME_REDUCTION), 
                  (255, 0, 255), 2)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        hand = result.hand_landmarks[0]
        
        # Teken alle landmarks
        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Highlight wijsvinger en duim
        index_x = int(hand[INDEX_TIP].x * w)
        index_y = int(hand[INDEX_TIP].y * h)
        thumb_x = int(hand[THUMB_TIP].x * w)
        thumb_y = int(hand[THUMB_TIP].y * h)
        
        # Bereken pinch afstand voor visuele feedback
        pinch_distance = get_distance(hand[THUMB_TIP], hand[INDEX_TIP])
        is_pinching = pinch_distance < 0.06
        
        # Kleur verandert op basis van pinch status
        line_color = (0, 255, 0) if is_pinching else (255, 255, 0)
        line_thickness = 3 if is_pinching else 2
        
        cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), 2)
        cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), 2)
        cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), line_color, line_thickness)
        
        # Toon pinch afstand
        cv2.putText(frame, f"Pinch: {pinch_distance:.3f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detecteer gebaar
        gesture = detect_gesture_mode(hand)
        
        # Verwerk gebaar
        if gesture == "MOVE":
            # Converteer hand positie naar schermcoördinaten
            # Gebruik alleen actieve zone
            x_norm = (hand[INDEX_TIP].x * w - FRAME_REDUCTION) / (w - 2 * FRAME_REDUCTION)
            y_norm = (hand[INDEX_TIP].y * h - FRAME_REDUCTION) / (h - 2 * FRAME_REDUCTION)
            
            # Begrens tussen 0 en 1
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            
            # Pas sensitiviteit toe - van centrum uit
            x_center = 0.5
            y_center = 0.5
            x_norm = x_center + (x_norm - x_center) * SENSITIVITY
            y_norm = y_center + (y_norm - y_center) * SENSITIVITY
            
            # Opnieuw begrenzen na sensitivity
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            
            # Converteer naar scherm pixels
            screen_x = int(x_norm * screen_w)
            screen_y = int(y_norm * screen_h)
            
            # Smoothing toepassen
            if prev_x != 0:
                screen_x = int(prev_x * SMOOTHING + screen_x * (1 - SMOOTHING))
                screen_y = int(prev_y * SMOOTHING + screen_y * (1 - SMOOTHING))
            
            prev_x, prev_y = screen_x, screen_y
            
            # Beweeg muis
            pyautogui.moveTo(screen_x, screen_y, duration=0)
            
            # Reset scroll start
            scroll_start_y = None
            
            cv2.putText(frame, "MUIS BEWEGEN", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        elif gesture == "SCROLL":
            # Track verticale beweging voor scrollen
            current_y = hand[INDEX_TIP].y
            
            if scroll_start_y is None:
                scroll_start_y = current_y
            else:
                # Bereken verschil
                y_diff = scroll_start_y - current_y
                
                # Als significante beweging, scroll
                if abs(y_diff) > 0.02:  # Drempelwaarde
                    scroll_amount = int(y_diff * 500)  # Vermenigvuldiger voor gevoeligheid
                    pyautogui.scroll(scroll_amount)
                    scroll_start_y = current_y
            
            cv2.putText(frame, "SCROLLEN", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)
        
        elif gesture == "CLICK" and click_cooldown == 0:
            current_time = time.time()
            
            # Check voor dubbele klik (binnen 0.5 seconden)
            if current_time - last_click_time < 0.5:
                pyautogui.doubleClick()
                last_click_time = 0  # Reset om triple clicks te voorkomen
            else:
                pyautogui.click()
                last_click_time = current_time
            
            click_cooldown = 15  # Voorkom dubbele clicks
            scroll_start_y = None
            cv2.putText(frame, "KLIK!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        elif gesture == "RIGHT_CLICK" and right_click_cooldown == 0:
            pyautogui.rightClick()
            right_click_cooldown = 15
            scroll_start_y = None
            cv2.putText(frame, "RECHTER KLIK!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        elif gesture == "PAUSE":
            scroll_start_y = None
            cv2.putText(frame, "GEPAUZEERD", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        else:
            scroll_start_y = None
            cv2.putText(frame, "GEREED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Toon extra info
        fingers = count_fingers(hand)
        cv2.putText(frame, f"Vingers: {fingers}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:
        cv2.putText(frame, "GEEN HAND GEDETECTEERD", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Cooldown verlagen
    if click_cooldown > 0:
        click_cooldown -= 1
    if right_click_cooldown > 0:
        right_click_cooldown -= 1
    
    # FPS berekening
    fps_counter += 1
    if time.time() - fps_start_time > 1:
        current_fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # Toon FPS
    cv2.putText(frame, f"FPS: {current_fps}", (w - 120, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Hand Muisbesturing", frame)
    timestamp += 33  # ~30 FPS

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    
    # OPLOSSING: Check of window was gesloten met X-knop
    try:
        if cv2.getWindowProperty("Hand Muisbesturing", cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()
print("\nMuisbesturing gestopt!")