import pybullet as pb
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from robot_driving import *
from corners_detection import *


########################################################################
# Подключаем мир, робота, делаем первичные настройки
########################################################################

# pb.connect(pb.GUI)
pb.connect(pb.DIRECT)

pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)

# Загружаем плоскость
pb.loadURDF("plane.urdf")

# Загружаем робота
obj = pb.loadURDF("diff_drive.urdf.xml", -4, -4, 0.1)

# Загружаем стены
wallsId = pb.loadURDF(
    "walls.urdf.xml",
    useFixedBase=True
)

# Настраиваем камеру, с помощью которой смотрим 
# на визуальное отображение сцены
# pb.resetDebugVisualizerCamera(
#     cameraDistance=7.0,
#     cameraYaw=250,
#     cameraPitch=-90,
#     cameraTargetPosition=[0, 0, 0.5]
# )


########################################################################
# Задаём параметры карты и движения робота
########################################################################

MAP_WIDTH = 12
MAP_HEIGHT = 12
RESOLUTION = 1

grid = np.zeros(
    (
        int(MAP_WIDTH / RESOLUTION),
        int(MAP_HEIGHT / RESOLUTION)
    )
)

# wheel order: [right, left, right, left]
motorIdx = [0, 1, 2, 3]

SPEED = 8.0
d = 2.0

DISTANCE_FRONTSIDE = 2.83
DISTANCE_TURN = 2.5
DISTANCE_STOP = 0.7
DISTANCE_SIDE = 0.9


########################################################################
# Задаём углы и цвет лучей лидара для управления движением робота
########################################################################

angles_local = [
    np.pi / 2,
    np.pi / 4,
    0,
    -np.pi,
    3 * np.pi / 4
]

ray_colors = [
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [0, 1, 1]   # lightblue
]


########################################################################
# Запускаем робота
########################################################################

# общее время проезда робота
maxTime = 890
# время перехода от проезда по периметру помещения к змейке
midTime = 460

dt = 1 / 60

logTime = np.arange(0, maxTime, dt)

RAY_LENGTH = 10.0
RAY_NUMBER = 180

robot_positions = []
lidar_points = []
all_observed_corners = []

for t in logTime:

    # ground truth
    pos, orn = pb.getBasePositionAndOrientation(obj)

    y = pos[1]

    euler = pb.getEulerFromQuaternion(orn)
    angle = euler[2]

    robot_positions.append([pos[0], pos[1]])

    ####################################################################
    # Пускаем 360 лучей для поиска углов
    ####################################################################

    ray_angles = np.linspace(0.0, 2 * np.pi, RAY_NUMBER)

    pos_from = np.tile(
        [pos[0], pos[1], pos[2] + 0.1],
        (RAY_NUMBER, 1)
    )

    ray_end_x = pos[0] + RAY_LENGTH * np.cos(ray_angles)
    ray_end_y = pos[1] + RAY_LENGTH * np.sin(ray_angles)
    ray_end_z = np.full(RAY_NUMBER, pos[2] + 0.1)

    pos_to = np.vstack(
        (ray_end_x, ray_end_y, ray_end_z)
    ).T

    results = pb.rayTestBatch(
        pos_from,
        pos_to
    )

    hit_fractions = [ray[2] for ray in results]

    distances = np.array(hit_fractions) * RAY_LENGTH

    hit_positions = [ray[3] for ray in results]

    lidar_points_step = []

    merge_distance = 0.1

    for i, hit_fraction in enumerate(hit_fractions):

        if hit_fraction < 1.0:

            lidar_points_step.append(
                [
                    hit_positions[i][0],
                    hit_positions[i][1]
                ]
            )

    lidar_points.extend(lidar_points_step)

    all_observed_corners.extend(
        np.array(
            detect_corners(
                lidar_points_step,
                pos[0],
                pos[1],
                all_observed_corners
            )
        )
    )

    ####################################################################
    # Пускаем 5 лучей для управления роботом
    ####################################################################

    ray_angles = [
        angle + np.pi / 2,
        angle + np.pi / 4,
        angle,
        angle - np.pi,
        angle + 3 * np.pi / 4
    ]

    pos_from = np.tile(
        [pos[0], pos[1], pos[2] + 0.1],
        (5, 1)
    )

    ray_end_x = pos[0] + RAY_LENGTH * np.cos(ray_angles)
    ray_end_y = pos[1] + RAY_LENGTH * np.sin(ray_angles)
    ray_end_z = np.full(5, pos[2] + 0.1)

    pos_to = np.vstack(
        (ray_end_x, ray_end_y, ray_end_z)
    ).T

    results = pb.rayTestBatch(
        pos_from,
        pos_to
    )

    hit_fractions = [ray[2] for ray in results]

    distances = np.array(hit_fractions) * RAY_LENGTH

    dist_0, dist_45, dist_90, dist_270, dist_315 = distances

    # Выбираем режим движения
    if t < midTime:

        # Сначала обходим внешние стены
        speeds = calculate_wheel_speeds_square(
            dist_0,
            dist_45,
            dist_90,
            dist_270,
            dist_315,
            y
        )

    else:

        # Потом едем змейкой
        speeds = calculate_wheel_speeds_snake(
            dist_0,
            dist_45,
            dist_90,
            dist_270,
            dist_315,
            y
        )

    # Применяем скорости
    pb.setJointMotorControlArray(
        bodyIndex=obj,
        jointIndices=motorIdx,
        targetVelocities=speeds,
        controlMode=pb.VELOCITY_CONTROL,
        forces=[100.0, 100.0, 100.0, 100.0]
    )

    print(
        f"Время: {t:.1f}с, "
        f"Позиция: ({pos[0]:.2f}, {pos[1]:.2f}), "
        f"Расстояния: "
        f"{dist_0:.2f}, "
        f"{dist_45:.2f}, "
        f"{dist_90:.2f}, "
        f"{dist_270:.2f}, "
        f"{dist_315:.2f}, "
        f"Скорости: "
        f"{speeds[0]:.1f}, "
        f"{speeds[1]:.1f}, "
        f"{speeds[2]:.1f}, "
        f"{speeds[3]:.1f}"
    )
    
    pb.stepSimulation()


# Преобразуем списки в numpy arrays

robot_positions = np.array(robot_positions)

lidar_points = np.array(lidar_points)

all_observed_corners = (
    np.array(all_observed_corners)
    if all_observed_corners
    else np.array([])
)


########################################################################
# Отрисовываем график движения
########################################################################

plt.figure(figsize=(10, 8))

plt.scatter(
    lidar_points[:, 0],
    lidar_points[:, 1],
    c='#099cd6',
    s=5,
    alpha=0.3,
    label='Стены'
)

if len(all_observed_corners) > 0:

    plt.scatter(
        all_observed_corners[:, 0],
        all_observed_corners[:, 1],
        c='red',
        s=80,
        marker='x',
        label='Найденные углы'
    )

plt.plot(
    robot_positions[:, 0],
    robot_positions[:, 1],
    color="#c94016",
    linestyle="-",
    linewidth=2,
    label='Путь робота'
)

plt.scatter(
    robot_positions[0, 0],
    robot_positions[0, 1],
    c='#194f28',
    s=100,
    marker='o',
    label='Начало'
)

plt.scatter(
    robot_positions[-1, 0],
    robot_positions[-1, 1],
    c='#e0a016',
    s=100,
    marker='o',
    label='Конец'
)

plt.xlabel('X')
plt.ylabel('Y')

plt.title('Путь робота и стены')

plt.legend(loc='upper right')

plt.gca().xaxis.set_major_locator(
    ticker.MultipleLocator(1)
)

plt.gca().yaxis.set_major_locator(
    ticker.MultipleLocator(1)
)

plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()