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

# --- Fun extras -------------------------------------------------------------
# Reaction-time ratings shown after a round (seconds, upper bound for each).
RATING_PERFECT = 0.6
RATING_GREAT = 1.0
RATING_GOOD = 2.0          # slower than this but still in time = "OK"
# Pose Battle rates accuracy instead: % needed for PERFECT, GREAT, GOOD.
ACCURACY_RATINGS = (95, 88, 81)

# Rounds won in a row before a player is "on fire".
STREAK_ON_FIRE = 3

# Confetti
CONFETTI_BURST = 140       # particles per burst
CONFETTI_MAX = 700         # cap so the frame rate never suffers

# Best-moment photos: how many recent frames to keep. A round ends a few
# frames after the reaction is confirmed, so a short buffer is enough.
MOMENT_BUFFER_FRAMES = 10

# --- Hold Challenge ---------------------------------------------------------
HOLD_TARGET = 6.0          # holding this long earns full points
HOLD_TIME_LIMIT = 10.0     # round length (seconds after GO)
# Losing the pose for up to this long doesn't break the hold, so a single
# missed detection or a wobble doesn't end it.
HOLD_DROPOUT = 0.3

# --- Pose similarity / Pose Battle -----------------------------------------
# A joint scores 100% when its angle matches the target exactly, falling
# linearly to 0% at this many degrees off.
SIM_MAX_DIFF = 60.0
SIM_MIN_JOINTS = 6         # need at least this many of the 8 joints visible
POSE_MATCH_THRESHOLD = 75  # % similarity that counts as "matching"
POSE_CONFIRM_FRAMES = 5    # must match this many frames in a row
POSE_TIME_LIMIT = 8.0
POSE_ACCURACY_MAX = 50     # extra points for accuracy above the threshold

# --- Objects / Bring to Camera (Solo only) ---------------------------------
OBJECT_MODEL = "yolov8n.pt"   # pretrained on COCO, downloaded on first use
OBJECT_CLASSES = ("bottle", "cup", "book", "cell phone")
OBJECT_MIN_CONFIDENCE = 0.35
OBJECT_IMG_SIZE = 480         # YOLO input size; smaller = faster
# "Close to the camera" = the object's box covers at least this share of the
# frame. Phones are small, so they get a lower bar.
OBJECT_CLOSE_AREA = {"bottle": 0.08, "cup": 0.06, "book": 0.12, "cell phone": 0.04}
OBJECT_CONFIRM_FRAMES = 3     # must stay close this many frames in a row
OBJECT_TIME_LIMIT = 12.0
OBJECT_FAST = 3.0             # bring it within this many seconds for the full bonus

# --- Timers get shorter in later rounds -------------------------------------
# Each round's time limit is scaled by (1 - SHRINK * round index), never
# below MIN_SCALE. With 5 rounds: 100%, 92.5%, 85%, 77.5%, 70%.
TIMER_SHRINK_PER_ROUND = 0.075
TIMER_MIN_SCALE = 0.7

# --- Voice ------------------------------------------------------------------
VOICE_RATE = 190              # words per minute
VOICE_MAX_QUEUE = 3           # older lines are dropped so speech never lags the game
