import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# For mouse control
screen_w, screen_h = pyautogui.size()
prev_pinched = False
pinch_blinks = []
blink_time_window = 0.8  # seconds
last_mouse_move = 0
mouse_move_delay = 0.01



# Gesture helpers for direction
def fingers_up(lm_list):
    # Returns a list of bools for [thumb, index, middle, ring, pinky]
    if len(lm_list) < 21:
        return [False]*5
    up = []
    # Thumb: tip_x > ip_x (right hand)
    _, thumb_tip_x, _ = lm_list[4]
    _, thumb_ip_x, _ = lm_list[3]
    up.append(thumb_tip_x > thumb_ip_x)
    # Other fingers: tip_y < pip_y
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        _, _, tip_y = lm_list[tip]
        _, _, pip_y = lm_list[pip]
        up.append(tip_y < pip_y)
    return up

def is_thumb_index_pinch(lm_list, threshold=40):
    # Returns True if thumb tip and index tip are close
    if len(lm_list) < 9:
        return False
    _, x1, y1 = lm_list[4]  # Thumb tip
    _, x2, y2 = lm_list[8]  # Index tip
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return dist < threshold

def is_index_up_only(lm_list):
    # Returns True if only index finger is up (excluding thumb)
    if len(lm_list) < 21:
        return False
    up = []
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for i, (tip, pip) in enumerate(zip(tips, pips)):
        _, _, tip_y = lm_list[tip]
        _, _, pip_y = lm_list[pip]
        up.append(tip_y < pip_y)
    return up[0] and not any(up[1:])

def is_hand_open(lm_list):
    # All four fingers (except thumb) are open
    if len(lm_list) < 21:
        return False
    fingers = []
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        _, _, tip_y = lm_list[tip]
        _, _, pip_y = lm_list[pip]
        fingers.append(tip_y < pip_y)
    # Thumb: tip_x > ip_x for right hand, < for left hand (simple version: tip_x > pip_x)
    _, thumb_tip_x, _ = lm_list[4]
    _, thumb_ip_x, _ = lm_list[3]
    thumb_open = thumb_tip_x > thumb_ip_x
    return all(fingers) and thumb_open

def is_fingers_up(lm_list, finger_ids):
    # Returns True if only the specified fingers are up (excluding thumb)
    if len(lm_list) < 21:
        return False
    up = []
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for i, (tip, pip) in enumerate(zip(tips, pips)):
        _, _, tip_y = lm_list[tip]
        _, _, pip_y = lm_list[pip]
        up.append(tip_y < pip_y)
    # Only the specified fingers are up
    return all(up[i] if i in finger_ids else not up[i] for i in range(4))

def get_cursor_pos(lm_list, frame_shape):
    # Map index tip to screen coordinates
    _, x, y = lm_list[8]
    h, w, _ = frame_shape
    screen_x = int(x / w * screen_w)
    screen_y = int(y / h * screen_h)
    return screen_x, screen_y




# Smoothing and deadzone for mouse movement
smooth_x, smooth_y = None, None
alpha = 0.15  # more smoothing
deadzone = 12  # pixels
click_hold_time = 0.18  # seconds to hold pinch before click
pinch_start_time = None

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Double tap (blink) index finger and thumb for click
            thumb_index_pinch = is_thumb_index_pinch(lm_list)
            if thumb_index_pinch and not prev_pinched:
                pinch_blinks.append(now)
                # Keep only last 2 blinks
                pinch_blinks = [t for t in pinch_blinks if now-t < blink_time_window]
                if len(pinch_blinks) >= 2:
                    pyautogui.click()
                    pinch_blinks = []
                prev_pinched = True
            elif not thumb_index_pinch:
                prev_pinched = False




            now = time.time()
            move_name = "NONE"
            move_dist = 40  # pixels per gesture
            up = fingers_up(lm_list)  # [thumb, index, middle, ring, pinky]

            # Thumb+Index: right, Thumb+Index+Middle: left, Only Index: up, Thumb+Pinky: down, Only Pinky: down
            if up == [1,1,0,0,0]:
                move_name = "RIGHT"
                if now - last_mouse_move > mouse_move_delay:
                    x, y = pyautogui.position()
                    pyautogui.moveTo(min(x+move_dist, screen_w-1), y)
                    last_mouse_move = now
            elif up == [1,1,1,0,0]:
                move_name = "LEFT"
                if now - last_mouse_move > mouse_move_delay:
                    x, y = pyautogui.position()
                    pyautogui.moveTo(max(x-move_dist, 0), y)
                    last_mouse_move = now
            elif up == [0,1,0,0,0]:
                move_name = "UP"
                if now - last_mouse_move > mouse_move_delay:
                    x, y = pyautogui.position()
                    pyautogui.moveTo(x, max(y-move_dist, 0))
                    last_mouse_move = now
            # Removed thumb + pinky for down. Only pinky alone moves down.
            elif up == [0,0,0,0,1]:
                move_name = "DOWN (PINKY)"
                if now - last_mouse_move > mouse_move_delay:
                    x, y = pyautogui.position()
                    pyautogui.moveTo(x, min(y+move_dist, screen_h-1))
                    last_mouse_move = now

            # Scrolling gestures
            # Open hand (all fingers up): scroll down
            # Closed hand (fist, all fingers down): stop scrolling
            # Index+middle up: scroll up
            scroll_name = None
            if up == [1,1,1,1,1]:
                scroll_name = "SCROLL DOWN"
                if now - last_mouse_move > mouse_move_delay:
                    pyautogui.scroll(-30)
                    last_mouse_move = now
            elif up == [0,1,1,0,0]:
                scroll_name = "SCROLL UP"
                if now - last_mouse_move > mouse_move_delay:
                    pyautogui.scroll(30)
                    last_mouse_move = now
            elif up == [0,0,0,0,0]:
                scroll_name = "STOP SCROLLING"
                # Do not scroll when fist is detected

            # Draw movement and scroll name
            color = (0,255,0) if move_name!="NONE" else (0,0,255)
            cv2.putText(frame, f"Move: {move_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if scroll_name:
                cv2.putText(frame, f"Scroll: {scroll_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,128,0), 2)

            # ...existing code for scrolling and landmarks...

            # Draw landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Scroll Gesture", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

