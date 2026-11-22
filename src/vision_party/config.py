"""All tunable thresholds in one place.

Distances are expressed as multiples of the player's torso length (shoulder
midpoint to hip midpoint, in pixels). That keeps every rule independent of how
far the player stands from the camera.
"""

# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- MediaPipe Pose ---------------------------------------------------------
MODEL_COMPLEXITY = 1          # 0 = fastest, 2 = most accurate
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MIN_VISIBILITY = 0.5          # a landmark below this is treated as missing

# --- Action rules -----------------------------------------------------------
# Hand "up": wrist must be this far above the nose (torso lengths).
HAND_UP_MARGIN = 0.10

# Squat: average knee angle below this (degrees) ...
SQUAT_KNEE_ANGLE = 115.0
# ... and the hips must have dropped: vertical hip-to-knee gap below this.
SQUAT_HIP_KNEE_GAP = 0.45

# One leg up: one ankle higher than the other by at least this much.
ONE_LEG_ANKLE_GAP = 0.30

# T-pose: elbows nearly straight, wrists near shoulder height, arms spread.
T_POSE_ELBOW_ANGLE = 150.0
T_POSE_WRIST_HEIGHT_TOL = 0.30
T_POSE_SPREAD = 2.2           # wrist-to-wrist distance / torso length

# --- Match flow (seconds) ---------------------------------------------------
ROUNDS_PER_MATCH = 5
INTRO_SECONDS = 2.0           # instruction shown (and later spoken) before countdown
COUNTDOWN_SECONDS = 3         # "3, 2, 1"
GO_FLASH_SECONDS = 0.6        # how long "GO!" stays on screen
RESULT_SECONDS = 3.0

# --- Scoring ----------------------------------------------------------------
BASE_POINTS = 100
SPEED_BONUS_MAX = 50

# --- Reaction Battle --------------------------------------------------------
REACTION_TIMEOUT = 5.0
# An action must hold for this many consecutive frames to count, so a single
# noisy frame can't win a round. Reaction time is taken from the first frame.
REACTION_CONFIRM_FRAMES = 3
REACTION_FAST = 0.4           # reaction time that earns the full speed bonus
