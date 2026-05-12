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

# Настраиваем камеру, с помощью которой смотрим 
# на визуальное отображение сцены
# pb.resetDebugVisualizerCamera(
#     cameraDistance=7.0,
#     cameraYaw=250,
#     cameraPitch=-90,
#     cameraTargetPosition=[5, 5, 0.5]
# )

pb.connect(pb.DIRECT)

pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)

# Загружаем плоскость
pb.loadURDF("plane.urdf")

# Загружаем робота
obj = pb.loadURDF("diff_drive.urdf.xml", 1, 1, 0.1)

# Загружаем стены
wallsId = pb.loadURDF(
    "walls.urdf.xml",
    useFixedBase=True
)


########################################################################
# Задаём константы
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

dt = 1 / 60

# общее время проезда робота
#maxTime = 880
maxTime = 500
# время перехода от проезда по периметру помещения к змейке
midTime = 460

logTime = np.arange(0, maxTime, dt)

joint_states = pb.getJointStates(obj, [0, 1, 2, 3])

ENCODER_NOISE = 0.0005
IMU_NOISE = 0.0005
RANGE_NOISE = 0.01
BEARING_NOISE = 0.005
# ENCODER_NOISE = 0.001
# IMU_NOISE = 0.001
# RANGE_NOISE = 0.01
# BEARING_NOISE = 0.01
# ENCODER_NOISE = 0
# IMU_NOISE = 0
# RANGE_NOISE = 0
# BEARING_NOISE = 0

def mahalanobis(z, z_hat, S):

    d = z - z_hat

    d[1] = np.arctan2(
        np.sin(d[1]),
        np.cos(d[1])
    )

    return d.T @ np.linalg.inv(S) @ d

########################################################################
# Запускаем робота
########################################################################

odometry_positions = []
ground_truth_positions = []
lidar_points = []
all_observed_corners = []
landmarks = []

last_angles = [
    joint_states[0][0],
    joint_states[1][0],
    joint_states[2][0],
    joint_states[3][0]
]

# задаём стартовое состояние
mu = np.array([1.0, 1.0, np.pi / 2]) # x, y, theta

Sigma = np.eye(3) * 0.01

for t in logTime:

    # ground truth
    pos, orn = pb.getBasePositionAndOrientation(obj)

    euler = pb.getEulerFromQuaternion(orn)
    angle = euler[2]

    ground_truth_positions.append([pos[0], pos[1]])

    ####################################################################
    # Работаем с одометрией
    ####################################################################

    joint_states = pb.getJointStates(obj, [0, 1, 2, 3])

    # Имитируем работу с энкодерами, получаем из них перемещение

    delta_angle_right = (
        (joint_states[0][0] - last_angles[0])
        + (joint_states[2][0] - last_angles[2])
    ) / 2 + np.random.normal(0, ENCODER_NOISE)

    delta_angle_left = (
        (joint_states[1][0] - last_angles[1])
        + (joint_states[3][0] - last_angles[3])
    ) / 2 + np.random.normal(0, ENCODER_NOISE)

    delta_distance_right = WHEEL_RADIUS * delta_angle_right
    delta_distance_left = WHEEL_RADIUS * delta_angle_left
    delta_distance = (delta_distance_right + delta_distance_left) / 2

    # Имитируем работу с IMU, получаем из него угол поворота

    physics_dt = pb.getPhysicsEngineParameters()["fixedTimeStep"]
    omega_z = pb.getBaseVelocity(obj)[1][2] + np.random.normal(0, IMU_NOISE)
    dtheta = omega_z * physics_dt

    # Обновляем положение робота

    x, y, theta = mu[:3]

    theta_mid = theta + dtheta / 2

    x += delta_distance * np.cos(theta_mid)
    y += delta_distance * np.sin(theta_mid)

    theta += dtheta
    theta = np.arctan2(np.sin(theta), np.cos(theta))

    mu[0] = x
    mu[1] = y
    mu[2] = theta

    n = len(mu)

    G = np.eye(n)

    G[0, 2] = -delta_distance * np.sin(theta_mid)
    G[1, 2] = delta_distance * np.cos(theta_mid)

    R3 = np.diag([
        ENCODER_NOISE**2 * abs(delta_distance),
        ENCODER_NOISE**2 * abs(delta_distance),
        IMU_NOISE**2
    ])

    R_full = np.zeros((n, n))

    R_full[:3, :3] = R3

    Sigma = G @ Sigma @ G.T + R_full

    odometry_positions.append(mu[:2].copy())

    last_angles = [
        joint_states[0][0],
        joint_states[1][0],
        joint_states[2][0],
        joint_states[3][0]
    ]

    ####################################################################
    # Пускаем RAY_NUMBER лучей для поиска углов-ориентиров
    ####################################################################

    ray_angles = np.linspace(0.0, 2 * np.pi, RAY_NUMBER)
    ray_angles_world = ray_angles + mu[2]

    pos_from = np.tile(
        [pos[0], pos[1], pos[2] + 0.1],
        (RAY_NUMBER, 1)
    )

    ray_end_x = mu[0] + RAY_LENGTH * np.cos(ray_angles_world)
    ray_end_y = mu[1] + RAY_LENGTH * np.sin(ray_angles_world)
    ray_end_z = np.full(RAY_NUMBER, pos[2] + 0.1)

    pos_to = np.vstack(
        (ray_end_x, ray_end_y, ray_end_z)
    ).T

    results = pb.rayTestBatch(
        pos_from,
        pos_to
    )

    hit_fractions = [ray[2] for ray in results]
    hit_positions = [ray[3] for ray in results]

    distances = np.array(hit_fractions) * RAY_LENGTH

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

    observed_corners = detect_corners(
        lidar_points_step,
        mu[0],
        mu[1],
        all_observed_corners
    )

    if len(observed_corners) > 0:
        all_observed_corners.extend(np.array(observed_corners))

    
    ####################################################################
    # EKF measurement update
    ####################################################################

    for corner in observed_corners:

        min_dist = float("inf")
        landmark_id = -1

        associated_data = None

        for i, lm in enumerate(landmarks):

            lm_x = mu[3 + 2 * i]
            lm_y = mu[3 + 2 * i + 1]

            dx = lm_x - mu[0]
            dy = lm_y - mu[1]

            q = dx**2 + dy**2

            if q < 1e-6:
                continue

            expected_range = np.sqrt(q)

            expected_bearing = (
                np.arctan2(dy, dx)
                - mu[2]
            )

            z = np.array([
                np.linalg.norm(
                    np.array(corner) - mu[:2]
                ),

                np.arctan2(
                    corner[1] - mu[1],
                    corner[0] - mu[0]
                ) - mu[2]
            ])

            z_hat = np.array([
                expected_range,
                expected_bearing
            ])

            n = len(mu)

            H = np.zeros((2, n))

            # robot part
            H[0, 0] = -dx / expected_range
            H[0, 1] = -dy / expected_range

            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1

            # landmark part
            lm_index = 3 + 2 * i

            H[0, lm_index] = dx / expected_range
            H[0, lm_index + 1] = dy / expected_range

            H[1, lm_index] = -dy / q
            H[1, lm_index + 1] = dx / q

            Q = np.diag([
                RANGE_NOISE**2,
                BEARING_NOISE**2
            ])

            S = H @ Sigma @ H.T + Q

            dist = mahalanobis(
                z,
                z_hat,
                S
            )

            if dist < min_dist:

                min_dist = dist

                landmark_id = i

                associated_data = (
                    H,
                    S,
                    z,
                    z_hat,
                    Q
                )

        ################################################################
        # New landmark
        ################################################################

        if (
            landmark_id == -1
            or min_dist > 5.99
        ):

            landmarks.append(corner)

            mu = np.hstack([
                mu,
                corner
            ])

            old_size = Sigma.shape[0]

            Sigma_new = np.zeros(
                (old_size + 2, old_size + 2)
            )

            Sigma_new[:old_size, :old_size] = Sigma

            Sigma_new[old_size:, old_size:] = (
                np.eye(2) * 1000
            )

            Sigma = Sigma_new

        ################################################################
        # Existing landmark
        ################################################################

        else:

            H, S, z, z_hat, Q = associated_data

            innovation = z - z_hat

            if np.linalg.norm(innovation) > 1.0:
                continue

            innovation[1] = np.arctan2(
                np.sin(innovation[1]),
                np.cos(innovation[1])
            )

            K = Sigma @ H.T @ np.linalg.inv(S)

            mu = mu + K @ innovation

            mu[2] = np.arctan2(
                np.sin(mu[2]),
                np.cos(mu[2])
            )

            I = np.eye(len(mu))

            Sigma = (
                (I - K @ H)
                @ Sigma
                @ (I - K @ H).T
                + K @ Q @ K.T
            )
            

    ####################################################################
    # Пускаем 5 лучей для управления роботом
    ####################################################################

    ray_angles = [
        mu[2] + np.pi / 2,
        mu[2] + np.pi / 4,
        mu[2],
        mu[2] - np.pi,
        mu[2] + 3 * np.pi / 4
    ]

    pos_from = np.tile(
        [pos[0], pos[1], pos[2] + 0.1],
        (5, 1)
    )

    ray_end_x = mu[0] + RAY_LENGTH * np.cos(ray_angles)
    ray_end_y = mu[1] + RAY_LENGTH * np.sin(ray_angles)
    ray_end_z = np.full(5, pos[2] + 0.1)

    pos_to = np.vstack(
        (ray_end_x, ray_end_y, ray_end_z)
    ).T

    results = pb.rayTestBatch(
        pos_from,
        pos_to
    )

    hit_fractions = [ray[2] for ray in results]

    # Вывод на экран в визуальном отображении сцены лучей для управления
    # hit_positions = [ray[3] for ray in results]

    # for i, hit_fraction in enumerate(hit_fractions):

    #     if hit_fraction < 1.0:

    #         pb.addUserDebugLine(pos_from[i], hit_positions[i], ray_colors[i], lifeTime=0.1)

    #     else:

    #         pb.addUserDebugLine(pos_from[i], pos_to[i], [0.5, 0.5, 0.5], lifeTime=0.1)

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
            mu[1]
        )

    else:

        # Потом едем змейкой
        speeds = calculate_wheel_speeds_snake(
            dist_0,
            dist_45,
            dist_90,
            dist_270,
            dist_315,
            mu[1]
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

odometry_positions = np.array(odometry_positions)

ground_truth_positions = np.array(ground_truth_positions)

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
    ground_truth_positions[:, 0],
    ground_truth_positions[:, 1],
    color="#c94016",
    linestyle="-",
    linewidth=2,
    label='Путь робота'
)

plt.plot(
    odometry_positions[:, 0],
    odometry_positions[:, 1],
    color="yellow",
    linestyle="-",
    linewidth=2,
    label='Одометрия'
)

plt.scatter(
    ground_truth_positions[0, 0],
    ground_truth_positions[0, 1],
    c='#194f28',
    s=100,
    marker='o',
    label='Начало'
)

plt.scatter(
    ground_truth_positions[-1, 0],
    ground_truth_positions[-1, 1],
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