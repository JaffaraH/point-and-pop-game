# 🎯 Point and Pop

A webcam-based finger-tracking shooting game built with **Python**, **OpenCV**, **MediaPipe**, and **Pygame**.
Use your **index finger** to shoot lasers at falling balls.

# Point and Pop

**Point and Pop** is a webcam-based finger-tracking shooting game built with **Python**, **OpenCV**, **MediaPipe**, and **Pygame**.  
You’ll use your **index finger** to aim and shoot lasers at colorful falling balls. No mouse or keyboard required!

---
## Gameplay Overview

- Balls fall from the top of the screen.
- Point with your **index finger** to fire lasers.
- **Pop** as many balls as you can before they reach the bottom.
- Smaller balls = more points.
- Miss too many? It’s game over.

---
## Technologies Used

- `Python 3.10+`
- `OpenCV` – Used for webcam capture and frame flipping
- `MediaPipe` – Tracks hand landmarks to detect pointing gestures
- `Pygame` – Renders the game visuals, plays sounds, and manages the game loop

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
