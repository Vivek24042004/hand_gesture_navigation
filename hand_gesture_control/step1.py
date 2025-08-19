import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
# Improved: Increase detection and tracking confidence for better finger detection
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,  # default is 0.5
    min_tracking_confidence=0.8    # default is 0.5
)

def is_hand_open(lm_list):
    # Check if all four fingers (except thumb) are open
    fingers = []
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        _, _, tip_y = lm_list[tip]
        _, _, pip_y = lm_list[pip]
        fingers.append(tip_y < pip_y)
    return all(fingers)

mp_draw = mp.solutions.drawing_utils

# For mouse navigation demo
screen_w, screen_h = 1920, 1080  # Default, update if needed
try:
    import pyautogui
    screen_w, screen_h = pyautogui.size()
except:
    pass
smooth_x, smooth_y = None, None
alpha = 0.15
deadzone = 12

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert color format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)



    # Draw hand landmarks, show hand state, and track index finger tip
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
            hand_state = "OPEN" if is_hand_open(lm_list) else "CLOSED"
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Hand: {hand_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if hand_state=="OPEN" else (0,0,255), 2)

            # Mouse navigation demo: track index finger tip with smoothing and deadzone
            _, tip_x, tip_y = lm_list[8]
            # Map to screen
            screen_x = int(tip_x / w * screen_w)
            screen_y = int(tip_y / h * screen_h)
            if smooth_x is None or smooth_y is None:
                smooth_x, smooth_y = screen_x, screen_y
            else:
                smooth_x = int(alpha * screen_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * screen_y + (1 - alpha) * smooth_y)
            # Deadzone: only show if moved enough
            if np.linalg.norm([smooth_x - screen_x, smooth_y - screen_y]) > deadzone:
                cv2.circle(frame, (tip_x, tip_y), 15, (255, 0, 0), 3)
                # Uncomment to move mouse: pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
