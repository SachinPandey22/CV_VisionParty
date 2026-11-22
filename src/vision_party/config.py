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
