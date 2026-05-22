import numpy as np

RAY_LENGTH = 10.0
RAY_NUMBER = 180
WHEEL_RADIUS = 0.05

ray_colors = [
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [0, 1, 1]   # lightblue
]

# общее время проезда робота
maxTime = 870
# время перехода от проезда по периметру помещения к змейке
midTime = 460

ENCODER_NOISE = 0.0001
IMU_NOISE = 0.0005
RANGE_NOISE = 0.05
BEARING_NOISE = 0.03
LIDAR_NOISE = 0.0001


TRUE_LANDMARKS = np.array([
    [0.0, 0.0],
    [0.0, 10.0],
    [10.0, 10.0],
    [10.0, 0.0],
    [2.0, 2.0],
    [4.0, 2.0],
    [2.0, 8.0],
    [4.0, 8.0],
    [6.0, 2.0],
    [8.0, 2.0],
    [6.0, 4.0],
    [8.0, 4.0],
    [6.0, 6.0],
    [8.0, 6.0],
    [6.0, 8.0],
    [8.0, 8.0],
])