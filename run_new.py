import pybullet as pb
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

####################################
# Подключаем мир, робота, делаем первичные настройки
####################################

# pb.connect(pb.GUI)
pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)

# load the plane to stand onto
pb.loadURDF("plane.urdf")
# elevate robot so that the wheels are touching the plane
obj = pb.loadURDF("diff_drive.urdf.xml", -4, -4, 0.1)
wallsId = pb.loadURDF("walls.urdf.xml", useFixedBase=True)

pb.resetDebugVisualizerCamera(
    cameraDistance=7.0,      # Расстояние от камеры до цели
    cameraYaw=250,           # Горизонтальный угол (0-360 градусов)
    cameraPitch=-90,        # Вертикальный угол (-90 до 90, отрицательное - смотреть сверху)
    cameraTargetPosition=[0, 0, 0.5]  # Точка, на которую смотрит камера [x, y, z]
)

####################################
# Задаём параметры карты
####################################

MAP_WIDTH = 12
MAP_HEIGHT = 12
RESOLUTION = 1
grid = np.zeros((int(MAP_WIDTH/RESOLUTION), int(MAP_HEIGHT/RESOLUTION)))

####################################
# Задаём углы лучей лидара для управления движением робота
####################################

angles_local = [np.pi/2, np.pi/4, 0, -np.pi, 3 * np.pi/4]  # 0°, 45°, 90°, 270°, 315°

ray_colors = [
    [1, 0, 0],  # red 0°
    [0, 1, 0],  # green 45°
    [0, 0, 1],  # blue 90°
    [1, 1, 0],  # yellow 270°
    [0, 1, 1]   # lightblue 315°
]

####################################
# Управляем движением робота
####################################

# wheel order: [right, left, right, left]
motorIdx = [0, 1, 2, 3]

RAY_LENGTH = 10.0
SPEED = 50.0
d = 10.0
DISTANCE_FRONTSIDE = 2.83
DISTANCE_TURN = 2.0
DISTANCE_STOP = 0.7
DISTANCE_SIDE = 0.9

def calculate_wheel_speeds_square(dist_0, dist_45, dist_90, dist_270, dist_315, y):

    # поворачиваем в верхней и нижней части поля, чтобы ехать змейкой
    if dist_0 <= 3:
        return [d, SPEED, d, SPEED]

    # не врезаемся в левую стену
    if dist_270 < DISTANCE_SIDE and dist_315 < DISTANCE_FRONTSIDE \
        and abs(y) < 4 and dist_45 < 4 and dist_90 < 4:
        return [d, SPEED, d, SPEED]

    # не врезаемся в правую стену
    if dist_90 < DISTANCE_SIDE and dist_45 < DISTANCE_FRONTSIDE \
        and abs(y) < 4 and dist_315 < 4 and dist_270 < 4:
        return [SPEED, d, SPEED, d]
    
    # если не сработали краевые случаи, то просто едем вперёд
    return [SPEED, SPEED, SPEED, SPEED]

def calculate_wheel_speeds_snake(dist_0, dist_45, dist_90, dist_270, dist_315, y):

    # не врезаемся в стену впереди
    if dist_0 <= DISTANCE_STOP:
        if dist_90 >= dist_270:
            return [d, SPEED, d, SPEED]
        else:
            return [SPEED, d, SPEED, d]

    # поворачиваем в верхней и нижней части поля, чтобы ехать змейкой
    if y > 0 and dist_45 > DISTANCE_FRONTSIDE and (dist_0 <= 3 or abs(y) > 4):
        return [d, SPEED, d, SPEED]
    elif y < 0 and dist_315 > DISTANCE_FRONTSIDE and (dist_0 <= 3 or abs(y) > 4): 
        return [SPEED, d, SPEED, d]

    # не врезаемся в левую стену
    if dist_270 < DISTANCE_SIDE and dist_315 < DISTANCE_FRONTSIDE \
        and abs(y) < 4 and dist_45 < 4 and dist_90 < 4:
        return [d, SPEED, d, SPEED]

    # не врезаемся в правую стену
    if dist_90 < DISTANCE_SIDE and dist_45 < DISTANCE_FRONTSIDE \
        and abs(y) < 4 and dist_315 < 4 and dist_270 < 4:
        return [SPEED, d, SPEED, d]
    
    # если не сработали краевые случаи, то просто едем вперёд
    return [SPEED, SPEED, SPEED, SPEED]

####################################
# Запускаем робота
####################################

maxTime = 60 
dt = 1 / 60
logTime = np.arange(0, maxTime, dt)

robot_positions = []
lidar_points = []

#############################
# обходим внешние стены изнутри по квадрату
#############################

for t in logTime:

    pos, orn = pb.getBasePositionAndOrientation(obj)
    y = pos[1]
    euler = pb.getEulerFromQuaternion(orn)
    angle = euler[2]

    robot_positions.append([pos[0], pos[1]])
    
    # measure the distance to obstacles with rays

    ray_angles = [angle + np.pi/2, angle + np.pi/4, angle, angle - np.pi, angle + 3 * np.pi/4]

    pos_from = np.tile([pos[0], pos[1], pos[2] + 0.1], (5,1))

    ray_end_x = pos[0] + RAY_LENGTH * np.cos(ray_angles)
    ray_end_y = pos[1] + RAY_LENGTH * np.sin(ray_angles)
    ray_end_z = np.full(5, pos[2] + 0.1)
    pos_to = np.vstack((ray_end_x, ray_end_y, ray_end_z)).T

    results = pb.rayTestBatch(
        pos_from,
        pos_to
    )

    hit_fractions = [ray[2] for ray in results]
    distances = np.array(hit_fractions) * RAY_LENGTH

    hit_positions = [ray[3] for ray in results]

    for i, hit_fraction in enumerate(hit_fractions):
        if hit_fraction < 1.0:
            #pb.addUserDebugLine(pos_from[i], hit_positions[i], ray_colors[i], lifeTime=0.1)
            lidar_points.append([hit_positions[i][0], hit_positions[i][1]])
        #else:
            #pb.addUserDebugLine(pos_from[i], pos_to[i], [0.5, 0.5, 0.5], lifeTime=0.1)
    
    dist_0, dist_45, dist_90, dist_270, dist_315 = distances

    speeds = calculate_wheel_speeds_square(dist_0, dist_45, dist_90, dist_270, dist_315, y)
    
    pb.setJointMotorControlArray(
        bodyIndex=obj,
        jointIndices=motorIdx,
        targetVelocities=speeds,
        controlMode=pb.VELOCITY_CONTROL,
        forces=[100.0, 100.0, 100.0, 100.0]
    )
    
    print(f"Время: {t:.1f}с, Позиция: ({pos[0]:.2f}, {pos[1]:.2f}), "
            f"Расстояния: {dist_0:.2f}, {dist_45:.2f}, {dist_90:.2f}, {dist_270:.2f}, {dist_315:.2f}, "
            f"Скорости: {speeds[0]:.1f}, {speeds[1]:.1f}")
    
    pb.stepSimulation()

#############################
# обходим змейкой внутреннюю часть помещения
#############################

pb.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

for t in logTime:

    pos, orn = pb.getBasePositionAndOrientation(obj)
    y = pos[1]
    euler = pb.getEulerFromQuaternion(orn)
    angle = euler[2]

    robot_positions.append([pos[0], pos[1]])
    
    # measure the distance to obstacles with rays

    ray_angles = [angle + np.pi/2, angle + np.pi/4, angle, angle - np.pi, angle + 3 * np.pi/4]

    pos_from = np.tile([pos[0], pos[1], pos[2] + 0.1], (5,1))

    ray_end_x = pos[0] + RAY_LENGTH * np.cos(ray_angles)
    ray_end_y = pos[1] + RAY_LENGTH * np.sin(ray_angles)
    ray_end_z = np.full(5, pos[2] + 0.1)
    pos_to = np.vstack((ray_end_x, ray_end_y, ray_end_z)).T

    results = pb.rayTestBatch(
        pos_from,
        pos_to
    )

    hit_fractions = [ray[2] for ray in results]
    distances = np.array(hit_fractions) * RAY_LENGTH

    hit_positions = [ray[3] for ray in results]

    for i, hit_fraction in enumerate(hit_fractions):
        if hit_fraction < 1.0:
            #pb.addUserDebugLine(pos_from[i], hit_positions[i], ray_colors[i], lifeTime=0.1)
            lidar_points.append([hit_positions[i][0], hit_positions[i][1]])
        #else:
            #pb.addUserDebugLine(pos_from[i], pos_to[i], [0.5, 0.5, 0.5], lifeTime=0.1)
    
    dist_0, dist_45, dist_90, dist_270, dist_315 = distances

    speeds = calculate_wheel_speeds_snake(dist_0, dist_45, dist_90, dist_270, dist_315, y)
    
    pb.setJointMotorControlArray(
        bodyIndex=obj,
        jointIndices=motorIdx,
        targetVelocities=speeds,
        controlMode=pb.VELOCITY_CONTROL,
        forces=[100.0, 100.0, 100.0, 100.0]
    )
    
    print(f"Время: {t:.1f}с, Позиция: ({pos[0]:.2f}, {pos[1]:.2f}), "
            f"Расстояния: {dist_0:.2f}, {dist_45:.2f}, {dist_90:.2f}, {dist_270:.2f}, {dist_315:.2f}, "
            f"Скорости: {speeds[0]:.1f}, {speeds[1]:.1f}")
    
    pb.stepSimulation()

robot_positions = np.array(robot_positions)
lidar_points = np.array(lidar_points)

####################################
# Отрисовываем график движения
####################################

plt.figure(figsize=(10, 8))
plt.scatter(lidar_points[:, 0], lidar_points[:, 1], 
            c='#099cd6', s=5, alpha=0.3, label='Стены')
plt.plot(robot_positions[:, 0], robot_positions[:, 1], 
         color="#c94016", linestyle="-", linewidth=2, label='Путь робота')
plt.scatter(robot_positions[0, 0], robot_positions[0, 1], 
            c='#194f28', s=100, marker='o', label='Начало')
plt.scatter(robot_positions[-1, 0], robot_positions[-1, 1], 
            c='#e0a016', s=100, marker='o', label='Конец')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot\'s path and walls')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()