# AI Hand Gesture Translator (English & Sinhala)

A real-time Computer Vision application built with Python, OpenCV, and Google's MediaPipe. It tracks hand landmarks via webcam and translates specific finger gestures into dynamic HUD messages in both English and Sinhala (Singlish).

## 🚀 Features
*   **Real-Time AI:** Detects 21 hand landmarks instantly using the modern MediaPipe Tasks API.
*   **Hold-to-Type Logic:** Prevents accidental spamming by requiring the user to hold a gesture for 1.5 seconds (with a visual progress bar).
*   **Bilingual Support:** Outputs both English (e.g., "HELLO") and Singlish (e.g., "Ayubowan") text.
*   **8 Distinct Gestures:** Recognizes Fist, Open Palm, Peace, Thumbs Up/Down, Point Up, and I Love You.
*   **Sci-Fi UI:** A sleek, neon-accented heads-up display built entirely with OpenCV.

## 🛠️ Technologies Used
*   Python 3
*   OpenCV (`cv2`)
*   MediaPipe (`mediapipe.tasks`)

## ⚙️ How to Run Locally
1. Install the dependencies:
   ```bash
   pip install opencv-python mediapipe