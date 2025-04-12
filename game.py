"""
Point and Pop: A Finger-Tracking Shooting Game

This game uses real-time webcam input and hand tracking to let you control a laser with your index finger.
Shoot falling balls by pointing — no mouse, no keyboard, just gestures.

Built with:
- OpenCV: for accessing the webcam and processing video frames
- MediaPipe: for detecting hand landmarks (like index finger and wrist)
- Pygame: for drawing the game visuals, playing sound, and managing game state
- NumPy: for math operations and quick array manipulation
- Math, Time, Random: for physics, cooldowns, and randomness
- JSON/OS: for storing high scores and handling file paths

Controls:
- Point with your index finger to fire lasers
- Pop as many falling balls as possible before they reach the bottom

Date: April 2025
"""
import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import random
import json
import os
from pygame import mixer

class FingerShootingGame:
    def __init__(self):
        # ========== Game Window & State Settings ==========
        self.width, self.height = 1280, 720 # Set window size in pixels (width x height)
        
        # Game control flags
        self.game_active = False # True while game is running (starts inactive until SPACE is pressed)
        self.game_paused = False # True when game is paused (toggled with SPACE)
        self.game_over = False   # True after losing all lives (shows game over screen)
        
        # Player score and lives
        self.score = 0                            # Current score (increases by popping balls)
        self.high_score = self.load_high_score()  # Load saved high score from file (JSON)
        self.max_lives = 5                        # Total lives the player starts with 5
        self.lives = self.max_lives               # Current remaining lives (decreases if ball missed)
  
        # ========== Difficulty & Ball Spawn Control ==========
        self.difficulty_level = 1                 # Level increases every 500 points (affects speed/spawn rate)
        self.ball_speed_min = 3                   # Base minimum speed of falling balls
        self.ball_speed_max = 7                   # Base maximum speed of falling balls
        self.ball_spawn_rate = 60                 # Number of frames between spawns (~60 FPS = 1 spawn/sec)
        self.ball_spawn_counter = 0               # Counts up every frame to decide when to spawn the next ball

        # Color Defenitions (RGB)
        self.WHITE  = (255, 255, 255)             # For UI text and borders
        self.BLACK  = (0, 0, 0)                   # For background overlays or shading
        self.RED    = (255, 0, 0)                 # For error states like "Game Over"
        self.PURPLE = (128, 0, 128)               # Color of the laser beam
        self.YELLOW = (255, 255, 0)               # Highlight color (e.g. new high score)

    
        #========== Pygame Initialization & Audio ==========
        pygame.init()                                                    # Starts the Pygame engine (required for all visuals/input)
        mixer.init()                                                     # Initializes the Pygame audio system for sound playback
        
        self.screen = pygame.display.set_mode((self.width, self.height)) # Create the game window with our defined width and height
        pygame.display.set_caption("Point and Pop")                      # Set the window title shown in the top bar
        self.clock = pygame.time.Clock()                                 # Pygame clock helps control the frame rate (e.g., 60 FPS)
        
        self.font = pygame.font.SysFont("Arial", 32)                     # Regular UI font
        self.big_font = pygame.font.SysFont("Arial", 64)                 # Large title font
        
        self.load_audio()                                                # Loads background, laser, and fail sounds

       
        # ========== Camera & Hand Tracking  ========== 
        # These functions initialize the webcam and enable hand tracking

        self.setup_camera()         # Start capturing video from the webcam
        self.setup_hand_tracking()  # Activate MediaPipe's hand detection model

        # ========== Object Tracking For Game State ==========
        # These lists will keep track of all moving objects in the game:
        self.balls = []      # Holds all falling balls (targets)
        self.lasers = []     # Holds all active lasers fired by the player
        self.particles = []  # Holds temporary visual effects when a ball is popped

        # Variables to manage laser shooting:
        self.last_shot_time = 0     # Timestamp of the last laser fired
        self.shot_cooldown = 0.3    # Minimum delay between shots (in seconds)
        
        # Variables related to hand detection:
        self.pointing = False        # True when the player is pointing
        self.finger_tip_pos = None   # (x, y) position of the tip of the index finger
        self.finger_direction = None # Direction vector from wrist to fingertip  

    #========== Audio Loading ==========
    # Loads all sound files (music and effects) used during the game
    def load_audio(self):
        try:
            # Get the full path to the current folder (makes audio loading portable)
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
            # Define paths to audio files relative to script location
            background_music_path = os.path.join(base_dir, "arcadewave.wav")
            laser_sound_path = os.path.join(base_dir, "laser.wav")
            fail_sound_path  = os.path.join(base_dir, "Fail_Sound.wav")
            
            # Background Music -> Try to load background music. It loops quietly while you play.
            if os.path.exists(background_music_path):
                pygame.mixer.music.load(background_music_path)
                pygame.mixer.music.set_volume(0.3) # Lower volume so it's not too distracting
                self.background_music_loaded = True
            else:
                 # If not found, print a message and set a flag to False
                print(f"Background music not found: {background_music_path}")
                self.background_music_loaded = False

            # Laser Sound Effect -> This plays each time you shoot a laser.
            if os.path.exists(laser_sound_path):
                self.laser_sound = mixer.Sound(laser_sound_path)
            else:
                # # If the laser sound is missing, use a silent "dummy" sound so the game doesn't crash.
                print(f"Laser sound not found: {laser_sound_path}")
                self.laser_sound = mixer.Sound(buffer=np.zeros(22050, dtype=np.int16)) # ~0.5 sec of silence
            self.laser_sound.set_volume(0.4)

            #  Fail Sound Effect -> This plays when the game ends.
            if os.path.exists(fail_sound_path):
                self.fail_sound = mixer.Sound(fail_sound_path)
            else:
                  # Fallback if the fail sound file is missing 
                print(f"Fail sound not found: {fail_sound_path}")
                self.fail_sound = mixer.Sound(buffer=np.zeros(22050, dtype=np.int16))
            self.fail_sound.set_volume(0.7)
       
        # Catch any unexpected errors in loading sounds
        except Exception as e:
            # If anything goes wrong (e.g., bad file, permission error), use silent sounds
            print(f"Error loading sounds: {e}")
            self.laser_sound = mixer.Sound(buffer=np.zeros(22050, dtype=np.int16))
            self.fail_sound = mixer.Sound(buffer=np.zeros(22050, dtype=np.int16))

    # ========== Camera & Hand Tracking Setup ==========
    def setup_camera(self):
        # We're grabbing the webcam using OpenCV.
        # This gives us live video input so we can detect your hand in real time.
        self.cap = cv2.VideoCapture(0) # 0 = default webcam
        
        # Make sure the camera feed matches our game window size (1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def setup_hand_tracking(self):
        # We're now setting up MediaPipe's hand tracking module.
        # This will let us track up to ONE hand and detect key landmarks like
        # the index finger tip, wrist, and joints — which we’ll use to fire lasers.

        self.mp_hands = mp.solutions.hands # Load the hand tracking class

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # Real-time tracking across video frames
            max_num_hands=1,               # We're only tracking one hand
            min_detection_confidence=0.5,  # If confidence is below 50%, it won't detect a hand
            min_tracking_confidence=0.5)   # Keeps tracking the hand if it's confident enough

    # =========== High Score Handling ===========
    def load_high_score(self):
        """
        Checks if a high_score.json file exists.
        If it does, we read the saved high score from it.
        If it doesn’t exist or something goes wrong, we return 0 (start fresh).
        """
        # Check if a high score file already exists
        # NOTE: This function doesn't create the fileThe file gets created the first time save_high_score() is called.
        try:
            if os.path.exists('high_score.json'):
                # Open the file and load the saved score
                with open('high_score.json', 'r') as f:
                    return json.load(f).get('high_score', 0) # Default to 0 if key not found
        except Exception as e:
            print(f"Error loading high score: {e}") # Print error but don’t crash
        return 0 # Return 0 if the file doesn't exist or failed to load

    def save_high_score(self):
        """
        Writes the current high score to a file so it’s saved for next time.
        We store it in a JSON file called high_score.json.
        """
        try:
            with open('high_score.json', 'w') as f:
                json.dump({'high_score': self.high_score}, f)
        except Exception as e:
            print(f"Error saving high score: {e}") # If it fails, just print the error

    # ===========  Game Object Creation =========== 
    def create_ball(self):
        """
        Spawns a new falling ball at a random x-position just above the screen.

        Each ball has:
        - a random size (which affects score and difficulty)
        - a random downward speed (scaled by difficulty level)
        - a color (just for fun visuals)
        - a 'wobble' effect to make its movement less robotic
        """
    
        size = random.randint(30, 70)
        ball = {
            'x': random.randint(size, self.width - size), # Random horizontal position
            'y': -size,  # Start above the screen
            'radius': size,
            'speed': random.uniform(self.ball_speed_min, self.ball_speed_max) * (1 + 0.2 * (self.difficulty_level - 1)),
            'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
            'wobble': random.uniform(-1, 1)  # Horizontal sway to give natural motion
        }
        self.balls.append(ball)  # Add the new ball to the active list

    def create_laser(self, start_pos, direction):
        """
        Fires a laser from the finger tip in the pointing direction.

        Only fires if enough time has passed since the last shot
        (to prevent laser spam — like a 'cooldown').

        Each laser moves in a straight line and disappears after a short life.
        """
        # Only allow firing if cooldown has passed
        if time.time() - self.last_shot_time < self.shot_cooldown:
            return
        
        # Normalize the direction vector (make it length 1)  so it moves consistently
        length = math.sqrt(direction[0]**2 + direction[1]**2) # Pythagorean theorem
        if length > 0:
            direction = (direction[0] / length, direction[1] / length) # unit vector
        
        # Create a dictionary representing a laser-> This tracks where it starts, its direction, speed, color, etc.
        laser = {
            'start_x': start_pos[0],
            'start_y': start_pos[1],
            'current_x': start_pos[0],
            'current_y': start_pos[1],
            'direction': direction,      # Normalized unit vector (Which way it should travel)
            'speed': 20,                 # How fast the laser moves per frame
            'color': self.PURPLE,        # What color it looks like when drawn
            'width': 5,                  # How thick the laser beam is
            'length': 20,                # How long the beam stretches backward (for visuals)
            'life': 30                   # How many frames the laser exists before disappearing
        }

        self.lasers.append(laser)                    # Add the laser to our list so it gets drawn and updated in the game.
        self.last_shot_time = time.time()            # Set the time of the last shot so we can check cooldown next time.
        self.laser_sound.play(loops=0, maxtime=2000) # Play a laser sound for feedback.

    def create_particles(self, x, y, color, count=20):
        """
        Spawns a small burst of particle effects when a ball is hit.

        Each particle:
        - Starts at the location where the ball was popped (x, y)
        - Has a random direction and speed (to scatter outward)
        - Is tiny, colorful, and disappears over time (like a spark)
        
        This is purely a visual effect — it doesn't affect gameplay, but it adds
        polish and feedback for the player, which is important in both casual and
        professional games for making the game feel satisfying and responsive.
        """
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi) # Random angle in radians (like spinning a wheel) -> his decides which direction the particle will travel in.
            speed = random.uniform(2, 8)           # Random speed so particles spread at different velocities
            size = random.randint(3, 8)            # Random size of the particle (adds variety, feels more natural)
            life = random.randint(20, 40)          # Random lifespan (how long the particle exists before fading out)
            
            #Create the particle as a dictionary 
            particle = {
                'x': x,                        # Start at the hit location (e.g., where the ball popped)
                'y': y,
                'dx': math.cos(angle) * speed, # X movement based on angle
                'dy': math.sin(angle) * speed, # Y movement based on angle
                'size': size,
                'color': color,                # Use the same color as the ball that was hitherits ball's color (a tuple of ints)
                'life': life,                  # Remaining time before it disappears
                'max_life': life               # Used to fade it out smoothly later
            }
            # Add this particle to the game’s active particle list
            self.particles.append(particle)

    # =========== Game Object Updates ===========
    def update_balls(self):
        """
        Moves each ball downward every frame.
        Adds a slight side-to-side wobble to make it feel less robotic.
        If a ball falls below the bottom of the screen, the player loses a life.
        """
        for ball in self.balls[:]: # Iterate over a shallow copy to safely remove during loop
            ball['y'] += ball['speed'] # Move the ball downward
            
            # Add horizontal wobble using sine wave (for more natural motion)
            ball['x'] += math.sin(time.time() * 2 + ball['wobble']) * 1.5
           
            # If the ball has passed the bottom edge of the screen
            if ball['y'] > self.height + ball['radius']:
                self.balls.remove(ball)      # Remove the missed ball
                self.lives -= 1              # Lose a life
                if self.lives <= 0 and not self.game_over:
                    self.trigger_game_over() # End game if out of lives

    def update_lasers(self):
        """
        Moves each laser forward in the direction it was fired.
        If a laser moves off-screen or lives too long, we remove it to clean up memory.
        """
        for laser in self.lasers[:]:
            # Move the laser in its normalized direction
            laser['current_x'] += laser['direction'][0] * laser['speed']
            laser['current_y'] += laser['direction'][1] * laser['speed']
            
            # Reduce remaining life of the laser
            laser['life'] -= 1
            
            # Remove laser if it goes off screen or its life runs out
            if (laser['current_x'] < 0 or laser['current_x'] > self.width or
                laser['current_y'] < 0 or laser['current_y'] > self.height or
                laser['life'] <= 0):
                self.lasers.remove(laser)

    def update_particles(self):
        """
        Updates the position of all particle effects.
        Simulates gravity by pulling particles downward slightly each frame.
        Removes the particle once its life runs out.
        """
        for particle in self.particles[:]:
            particle['x'] += particle['dx']         # Move horizontally
            particle['y'] += particle['dy']         # Move vertically
            particle['dy'] += 0.2                   # Simulate gravity by increasing downward velocity
            particle['life'] -= 1                   # Decrease life counter
            if particle['life'] <= 0:
                self.particles.remove(particle)     # Clean up when expired

    def check_collisions(self):
        """
        Checks for collisions between lasers and balls.
        If they collide:
        - Create particles at the impact point
        - Remove the laser and ball
        - Increase the score
        - Adjust difficulty if needed
       
     """
        for laser in self.lasers[:]:
            lx, ly = laser['current_x'], laser['current_y']
            for ball in self.balls[:]:
                 # Simple distance-based collision detection (circle vs. point)
                if math.hypot(lx - ball['x'], ly - ball['y']) < ball['radius']:
                    self.create_particles(ball['x'], ball['y'], ball['color'])
                    if laser in self.lasers:
                        self.lasers.remove(laser)
                    if ball in self.balls:
                        self.balls.remove(ball)
                    self.score += max(10, 100 - ball['radius']) # Smaller balls = higher points
                    self.update_difficulty() # Increase difficulty if score crosses threshold
                    break  # Only one ball hit per laser

    def update_difficulty(self):
        """
        Increases game difficulty based on player score.
        Every 500 points:
        - Balls fall faster
        - Balls spawn more frequently
        """
        new_level = 1 + self.score // 500
        if new_level > self.difficulty_level:
            self.difficulty_level = new_level
            self.ball_spawn_rate = max(30, 60 - (self.difficulty_level - 1) * 5)
            self.ball_speed_min = 3 + (self.difficulty_level - 1) * 0.5
            self.ball_speed_max = 7 + (self.difficulty_level - 1) * 0.8

    # ===========  Hand Gesture Detection (MediaPipe)  =========================
    def detect_hand_pose(self, frame):
        """
        Detects whether the player is making a pointing gesture.
        Uses index finger extension (tip far from base) and middle finger being *less* extended.
        If pointing, calculates direction and fires a laser from the fingertip.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Reset detection state for current frame
        self.pointing = False
        self.finger_tip_pos = None
        self.finger_direction = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                i_tip = (int(index_tip.x * w), int(index_tip.y * h))
                i_pip = (int(index_pip.x * w), int(index_pip.y * h))
                m_tip = (int(middle_tip.x * w), int(middle_tip.y * h))
                m_pip = (int(middle_pip.x * w), int(middle_pip.y * h))
                wrist_px = (int(wrist.x * w), int(wrist.y * h))

                # Measure how "extended" fingers are
                index_dist = math.hypot(i_tip[0] - i_pip[0], i_tip[1] - i_pip[1])
                middle_dist = math.hypot(m_tip[0] - m_pip[0], m_tip[1] - m_pip[1])

                # Pointing logic: index must be clearly more extended than middle
                if index_dist > 35 and index_dist > middle_dist * 1.2:
                    self.pointing = True
                    self.finger_tip_pos = i_tip
                    self.finger_direction = (
                        i_tip[0] - wrist_px[0],
                        i_tip[1] - wrist_px[1]
                    )

                # Draw landmarks regardless of detection
                self.draw_hand_landmarks(frame, hand_landmarks)

                # If pointing and game is active, fire laser
                if self.pointing and self.finger_tip_pos and self.finger_direction:
                    if self.game_active and not self.game_paused and not self.game_over:
                        self.create_laser(self.finger_tip_pos, self.finger_direction)

        return frame

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """
        Draws the hand skeleton (21 keypoints and connections) on the video frame.
        This helps the player see that hand tracking is working.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

    def distance(self, point1, point2):
        """
        Utility function: Calculates the straight-line distance between two (x, y) points.
        Used for gesture detection (e.g., finger extension).
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    # ===========  Rendering =========== 
    def render_game_objects(self, frame):
        """
        This function draws the real-time webcam feed, balls, lasers, and particle effects onto the game window.
        It gets called every frame (60x per second).
        """
        # Convert the OpenCV webcam frame (BGR) into a format Pygame can display.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)           # Convert BGR → RGB (OpenCV uses BGR, but Pygame uses RGB)
        frame = np.rot90(frame)                                  # Rotate frame for correct orientation
        surf = pygame.surfarray.make_surface(frame)              # Convert NumPy array to Pygame surface
        surf = pygame.transform.flip(surf, True, False)          # Mirror horizontally (makes hand movement intuitive)
        self.screen.blit(surf, (0, 0))                           # Draw the webcam background onto the screen
        
        # Draw falling balls
        for ball in self.balls:
            pygame.draw.circle(self.screen, ball['color'], (int(ball['x']), int(ball['y'])), int(ball['radius']))
        
        # Draw lasers (as short purple lines extending in the direction the player pointed)
        for laser in self.lasers:
            start_pos = (int(laser['current_x'] - laser['direction'][0] * laser['length']),
                         int(laser['current_y'] - laser['direction'][1] * laser['length']))
            end_pos = (int(laser['current_x']), int(laser['current_y']))
            pygame.draw.line(self.screen, laser['color'], start_pos, end_pos, laser['width'])
        
        # Draw explosion particle effects when a ball is hit
        for particle in self.particles:
            alpha = int(255 * (particle['life'] / particle['max_life']))
            # Explicitly cast each channel to int
            color = (int(particle['color'][0]), int(particle['color'][1]), int(particle['color'][2]), int(alpha))
            
            # Create a small transparent surface and draw a fading circle on it
            s = pygame.Surface((particle['size'] * 2, particle['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (particle['size'], particle['size']), particle['size'])
            self.screen.blit(s, (int(particle['x'] - particle['size']), int(particle['y'] - particle['size'])))

    def render_ui(self):
        """
        Renders UI text: score, high score, remaining lives, and level.
        Also calls the correct overlay screen depending on the game state.
        """
        self.screen.blit(self.font.render(f"Score: {self.score}", True, self.WHITE), (20, 20))
        self.screen.blit(self.font.render(f"High Score: {self.high_score}", True, self.WHITE), (20, 60))
        self.screen.blit(self.font.render(f"Lives: {self.lives}", True, self.WHITE), (self.width - 150, 20))
        self.screen.blit(self.font.render(f"Level: {self.difficulty_level}", True, self.WHITE), (self.width - 150, 60))
        
        # Display appropriate overlays depending on current game state
        if not self.game_active:
            self.render_start_screen()
        elif self.game_paused:
            self.render_pause_screen()
        elif self.game_over:
            self.render_game_over_screen()

    def render_start_screen(self):
        """
        Draws a translucent dark overlay with game title and basic instructions.
        Shown before the player presses SPACE to start.
        """
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("Point and Pop", True, self.WHITE)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 150))
        
        for i, text in enumerate([
            "Point your index finger to shoot lasers",
            "Pop the falling balls before they reach the bottom",
            "Press SPACE to start, ESC to quit"
        ]):
            self.screen.blit(self.font.render(text, True, self.WHITE),
                             (self.width // 2 - self.font.size(text)[0] // 2, 250 + i * 50))

    def render_pause_screen(self):
        """
        Shows a 'PAUSED' overlay when the player hits SPACE mid-game.
        Game state is frozen until SPACE is pressed again.
        """
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        pause = self.big_font.render("PAUSED", True, self.WHITE)
        self.screen.blit(pause, (self.width // 2 - pause.get_width() // 2, self.height // 2 - 100))

    def render_game_over_screen(self):
        """
        Final screen that appears when lives run out.
        Shows final score, high score, and prompts player to restart.
        """
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        over = self.big_font.render("GAME OVER", True, self.RED)
        self.screen.blit(over, (self.width // 2 - over.get_width() // 2, self.height // 2 - 150))
        
        self.screen.blit(self.font.render(f"Final Score: {self.score}", True, self.WHITE),
                         (self.width // 2 - self.font.size(f"Final Score: {self.score}")[0] // 2, self.height // 2 - 50))
        
        self.screen.blit(self.font.render(f"High Score: {self.high_score}", True, self.YELLOW),
                         (self.width // 2 - self.font.size(f"High Score: {self.high_score}")[0] // 2, self.height // 2))

    # =========== Game Sate Management ===========
    def trigger_game_over(self):
        """
        Called when the player runs out of lives.
        Freezes the game, plays fail sound, and checks for a new high score.
        """
        self.game_over = True       # Flag to indicate the game has ended
        pygame.mixer.music.pause()  # Pause background music
        self.fail_sound.play()      # Play fail sound (game over audio)
        
        # Clear remaining balls, lasers, and particles
        self.balls.clear()
        self.lasers.clear()
        self.particles.clear()
        
        # If the player's current score is higher than the previous high score,
        # update and save the new high score to file.
        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()

    def reset_game(self):
        """
        Resets the game state to default so the player can start a new game
        without restarting the entire program.
        """
        self.score = 0                      # Reset score to 0
        self.lives = self.max_lives         # Refill lives
        self.difficulty_level = 1           # Reset difficulty level
        self.ball_speed_min = 3             # Reset ball speed range to default
        self.ball_speed_max = 7
        self.ball_spawn_rate = 60           # Reset spawn rate of falling balls
        
        # Clear all active game objects
        self.balls = []                     # Remove all balls from screen
        self.lasers = []                    # Remove all lasers
        self.particles = []                 # Remove all visual effects
        self.game_over = False              # Game is no longer in a "game over" state

        pygame.mixer.music.unpause()  # Resume background music

    # ===========  Event Handling ===========
    def handle_events(self):
        """
        Monitors keyboard and window events (like quitting or pausing).
        This keeps the game interactive and responsive to user inputs.
        """

        for event in pygame.event.get():                    # Loop through all events
            if event.type == pygame.QUIT:
                return False                                # User clicked the X button — exit the game
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:            # ESC key pressed — exit the game
                    return False
                if event.key == pygame.K_SPACE:             # SPACE key triggers start, pause, resume, or restart depending on current game state
                    if not self.game_active:
                        self.game_active = True             # Start the game
                        if self.background_music_loaded:
                            pygame.mixer.music.play(-1)     # Loop music forever
                    elif self.game_paused:
                        self.game_paused = False            # Resume from pause
                        if self.background_music_loaded:
                            pygame.mixer.music.unpause()
                    elif self.game_over:
                        self.reset_game()                   # Reset all game state and start fresh
                        self.game_active = True
                    else:
                        self.game_paused = True             # Pause the game mid-play
                        if self.background_music_loaded:
                            pygame.mixer.music.pause()
        return True                                         # Keep running the game


    #  =========== Main Game Loop  ===========
    def run(self):
        """
        Main game loop — everything starts and runs here.
        This loop continuously updates game logic, reads input, draws visuals, and handles frame rate.
        """
        running = True
        while running:
            running = self.handle_events()                                              # Process keyboard/mouse/window events
            ret, frame = self.cap.read()                                                # Capture webcam frame
            if not ret:
                print("Error reading from webcam")
                break
            frame = cv2.flip(frame, 1)                                                  # Flip image horizontally for mirror view
            frame = self.detect_hand_pose(frame)                                        # Check if the player is pointing
            if self.game_active and not self.game_paused and not self.game_over:        # Game logic updates only if game is active and not paused or over
                self.ball_spawn_counter += 1
                if self.ball_spawn_counter >= self.ball_spawn_rate:
                    self.create_ball()                                                  # Add a new falling ball
                    self.ball_spawn_counter = 0
                # Update all objects (balls, lasers, particle effects)
                self.update_balls()
                self.update_lasers()
                self.update_particles()
                self.check_collisions()
           
            # Render the camera feed, game objects, and UI     
            self.render_game_objects(frame)
            self.render_ui()
            pygame.display.flip()                                                       # Push the drawn frame to the screen
            self.clock.tick(60)                                                         # Limit the frame rate to 60 FPS
        
        # Cleanup once game loop exits
        self.cap.release()                                                              # Release webcam
        self.hands.close()                                                              # Close MediaPipe hands
        pygame.quit()                                                                   # Close the game window and exit

if __name__ == "__main__":
    game = FingerShootingGame()
    game.run()
