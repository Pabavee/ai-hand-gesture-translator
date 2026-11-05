import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
from collections import Counter, deque

# --- CONFIGURATION ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading AI model... Please wait.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

# --- GESTURE LOGIC ---
def get_finger_states(lm):
    wrist = lm[0]
    palm = math.hypot(wrist.x - lm[9].x, wrist.y - lm[9].y) or 1e-6
    
    thumb = math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) / palm > 0.75
    
    def finger_up(tip, pip):
        return math.hypot(wrist.x - lm[tip].x, wrist.y - lm[tip].y) > math.hypot(wrist.x - lm[pip].x, wrist.y - lm[pip].y) * 1.05

    index = finger_up(8, 6)
    middle = finger_up(12, 10)
    ring = finger_up(16, 14)
    pinky = finger_up(20, 18)
    return [thumb, index, middle, ring, pinky]

def classify_gesture(lm):
    thumb, index, middle, ring, pinky = get_finger_states(lm)
    
    if all([thumb, index, middle, ring, pinky]): return "OPEN_PALM"
    if not any([thumb, index, middle, ring, pinky]): return "FIST"
    
    if thumb and not any([index, middle, ring, pinky]):
        dy = lm[4].y - lm[2].y
        if dy < 0: return "THUMBS_UP"
        else: return "THUMBS_DOWN"
        
    if thumb and index and pinky and not middle and not ring: return "I_LOVE_YOU"
    if index and middle and not ring and not pinky: return "PEACE"
    if index and not thumb and not middle and not ring and not pinky: return "POINT_UP"
    
    return None

# Map gestures to English & Singlish phrases
PHRASE_MAP = {
    "FIST": ("HELLO", "Ayubowan"),
    "OPEN_PALM": ("THANK YOU", "Sthuthiyi"),
    "PEACE": ("BYE", "Gihin ennam"),
    "I_LOVE_YOU": ("I LOVE YOU", "Mama oyata aadareyi"),
    "THUMBS_UP": ("SORRY", "Samawenna"),
    "POINT_UP": ("GOOD MORNING", "Subha udasanak"),
    "THUMBS_DOWN": ("GOOD NIGHT", "Subha rathriyak"),
}

PHRASE_COLORS = {
    "HELLO": (255, 200, 0),
    "THANK YOU": (0, 215, 255),
    "BYE": (255, 0, 255),
    "I LOVE YOU": (180, 105, 255),
    "SORRY": (0, 255, 0),
    "GOOD MORNING": (0, 165, 255),
    "GOOD NIGHT": (0, 0, 255),
}

# --- MAIN APP ---
def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1, 
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    history = deque(maxlen=7)
    prev_time = time.time()
    
    current_phrase = "" 
    current_color = (255,255,255)
    current_gesture = None
    hold_start_time = None
    HOLD_DURATION = 1.5 

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            raw_gesture = None
            
            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                h, w = frame.shape[:2]
                
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
                
                raw_gesture = classify_gesture(hand_lms)

            # Smoothing
            history.append(raw_gesture)
            votes = Counter(g for g in history if g)
            stable = None
            if votes:
                candidate, count = votes.most_common(1)[0]
                if count >= 4: stable = candidate

            now = time.time()
            h, w = frame.shape[:2]
            progress = 0.0

            # Typing logic
            if stable and stable in PHRASE_MAP:
                if stable != current_gesture:
                    current_gesture = stable
                    hold_start_time = now
                else:
                    elapsed = now - hold_start_time
                    progress = min(elapsed / HOLD_DURATION, 1.0)
                    
                    if elapsed >= HOLD_DURATION:
                        eng_text, si_text = PHRASE_MAP[stable]
                        current_phrase = (eng_text, si_text)
                        current_color = PHRASE_COLORS.get(eng_text, (255, 255, 255))
                        
                        hold_start_time = now + 10 
                        current_gesture = None
            else:
                current_gesture = None
                hold_start_time = None

            # --- DRAW UI ---
            cv2.rectangle(frame, (0, 0), (w, 40), (10, 10, 10), -1)
            cv2.putText(frame, "MODE: PHRASES (EN/SI)", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            
            fps = 1 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {int(fps)} | 'c': Clear | 'q': Quit", (w - 350, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            if current_gesture:
                bar_w = 300
                bar_h = 20
                bx = (w - bar_w) // 2
                by = 60
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * progress)
                
                color = PHRASE_COLORS.get(PHRASE_MAP[current_gesture][0], (255, 255, 255))
                
                cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), color, -1)
                cv2.putText(frame, "HOLD TO TYPE", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- BIG PHRASE DISPLAY (Center) ---
            if current_phrase:
                eng_text, si_text = current_phrase
                color = current_color
                
                # English Font Settings
                font_eng = cv2.FONT_HERSHEY_DUPLEX
                scale_eng = 2.5
                thick_eng = 4
                (w_eng, h_eng), _ = cv2.getTextSize(eng_text, font_eng, scale_eng, thick_eng)
                
                # Singlish Font Settings (Smaller)
                font_si = cv2.FONT_HERSHEY_SIMPLEX
                scale_si = 1.2
                thick_si = 2
                (w_si, h_si), _ = cv2.getTextSize(si_text, font_si, scale_si, thick_si)
                
                # Calculate center positions
                cx = (w - max(w_eng, w_si)) // 2
                cy = (h // 2) - 50
                
                # Dark background box
                padding = 30
                box_w = max(w_eng, w_si) + (padding * 2)
                box_h = h_eng + h_si + (padding * 3)
                bx = (w - box_w) // 2
                by = cy - h_eng - padding
                
                cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (20, 14, 10), -1)
                cv2.rectangle(frame, (bx, by), (bx + 10, by + box_h), color, -1)
                
                # Draw English
                cv2.putText(frame, eng_text, (cx + 3, cy + 3), font_eng, scale_eng, (0, 0, 0), thick_eng)
                cv2.putText(frame, eng_text, (cx, cy), font_eng, scale_eng, color, thick_eng)
                
                # Draw Singlish
                si_x = (w - w_si) // 2
                si_y = cy + h_si + 40
                cv2.putText(frame, si_text, (si_x + 2, si_y + 2), font_si, scale_si, (0, 0, 0), thick_si)
                cv2.putText(frame, si_text, (si_x, si_y), font_si, scale_si, (255, 255, 255), thick_si)

            cv2.imshow("AI Sign Language Translator", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_phrase = "" 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()