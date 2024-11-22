from pymunk import Vec2d

# Constant variables defined here
WIDTH, HEIGHT = 800, 600
CENTER = Vec2d(WIDTH / 2, HEIGHT / 2)
FPS = 60
RENDER_EVERY = 3

# Physics stuff
GROUND_HEIGHT = 50  # Prevent clipping
RAGDOLL_GROUP = 1

# AI parameters
SAVE_DIR = "models"
CONFIG_PATH = "./config.txt"
MAX_GENERATIONS = 1000
GENERATION_TICKS = FPS * 5
