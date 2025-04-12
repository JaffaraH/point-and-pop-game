# 🎯 Point and Pop

**Point and Pop** is a webcam-based finger-tracking shooting game built with **Python**, **OpenCV**, **MediaPipe**, and **Pygame**.  
Use your **index finger** to aim and shoot lasers at colorful falling balls — no mouse or keyboard required!

---
##  Gameplay Overview

- Balls fall from the top of the screen.
- Point your **index finger** to fire lasers.
- **Pop** as many balls as you can before they reach the bottom.
- Smaller balls = higher points.
- Lose all your lives? It’s game over.

---
## Features

-  **Hand Tracking**: Real-time finger detection using MediaPipe  
-  **Gesture-Based Shooting**: Fire by pointing your index finger  
-  **Audio Feedback**: Background music, laser fire, and fail sounds  
-  **Particle Effects**: Balls explode with satisfying visuals  
-  **Progressive Difficulty**: Faster spawn rates as your score increases  
-  **Score Tracking**: High score saved in `high_score.json`

---
## Project Structure
- `point_and_pop/`
  - `game.py` — Main game file  
  - `arcadewave.wav` — Background music  
  - `laser.wav` — Laser firing sound  
  - `Fail_Sound.wav` — Fail/game over sound  
  - `high_score.json` — Persistent high score file  
  - `README.md` — This file
    
---
## Technologies Used

- `Python 3.10+`
- `OpenCV` – Webcam capture and frame manipulation
- `MediaPipe` – Real-time hand and finger tracking
- `Pygame` – Graphics rendering, event handling, and audio

---
## How to Run the Game

1. Make sure you have **Python 3.10 or later** installed.
2. Install the required dependencies:

    ```bash
    pip install opencv-python mediapipe pygame numpy
    ```

3. Run the game:

    ```bash
    python game.py
    ```

- Your webcam will activate.  
- Point with your index finger to shoot!
