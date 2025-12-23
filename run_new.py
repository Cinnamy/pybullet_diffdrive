import pybullet as pb
import pybullet_data
import numpy as np
import time

pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)

# load the plane to stand onto
pb.loadURDF("plane.urdf")
# elevate robot so that the wheels are touching the plane
obj = pb.loadURDF("diff_drive.urdf.xml", -0.5, -0.5, 0.1)
wallsId = pb.loadURDF("walls.urdf.xml", useFixedBase=True)

pb.resetDebugVisualizerCamera(
    cameraDistance=3.0,      # Расстояние от камеры до цели
    cameraYaw=250,           # Горизонтальный угол (0-360 градусов)
    cameraPitch=-70,        # Вертикальный угол (-90 до 90, отрицательное - смотреть сверху)
    cameraTargetPosition=[0, 0, 0.5]  # Точка, на которую смотрит камера [x, y, z]
)

# wheel order: [right, left]
motorIdx = [0, 1]

# Параметры управления
RAY_LENGTH = 30.0
SPEED = 40.0
DISTANCE_45 = 1.4
DISTANCE_90 = 1.0

# Функция для расчета скоростей колес
def calculate_wheel_speeds(dist_0, dist_45, dist_90, dist_135):
    d = 10.0

    # [правое колесо, левое колесо]

    if dist_0 < DISTANCE_90 or dist_45 < DISTANCE_45:
        return [SPEED, d]

    if dist_90 == 30.0 or dist_45 == 30.0 or dist_90 > DISTANCE_45:
        return [d, SPEED]
    
    return [SPEED, SPEED]

# turn off castor wheel motor for free motion
pb.setJointMotorControl2(
    bodyIndex=obj,
    jointIndex=2,
    targetVelocity=0,
    controlMode=pb.VELOCITY_CONTROL,
    force=0,
)


maxTime = 20 
dt = 1 / 60
logTime = np.arange(0, maxTime, dt)

angles_local = [np.pi/2, np.pi/4, 0, -np.pi/4]  # 0°, 45°, 90°, 135°

ray_colors = [
    [1, 0, 0],  # red 0°
    [0, 1, 0],  # green 45°
    [0, 0, 1],  # blue 90°
    [1, 1, 0]   # yellow 135°
]

for t in logTime:

    pos, orn = pb.getBasePositionAndOrientation(obj)
    euler = pb.getEulerFromQuaternion(orn)
    yaw = euler[2]
    
    ray_distances = []
    ray_hit_points = []
    
    # measure the distance to obstacles with rays
    for i, angle_local in enumerate(angles_local):
        ray_angle = yaw + angle_local
        pos_from = [pos[0], pos[1], pos[2] + 0.1]
        pos_to = [
            pos[0] + RAY_LENGTH * np.cos(ray_angle),
            pos[1] + RAY_LENGTH * np.sin(ray_angle),
            pos[2] + 0.1
        ]
        
        ray_result = pb.rayTest(pos_from, pos_to)[0]
        hit_fraction = ray_result[2]
        hit_position = ray_result[3]
        
        distance = hit_fraction * RAY_LENGTH
        ray_distances.append(distance)
        ray_hit_points.append(hit_position)

        ray_end = [
            pos[0] + RAY_LENGTH * np.cos(ray_angle),
            pos[1] + RAY_LENGTH * np.sin(ray_angle),
            pos[2] + 0.1
        ]
        
        ray_result = pb.rayTest(pos_from, ray_end)[0]
        hit_fraction = ray_result[2]
        hit_position = ray_result[3]

        if hit_fraction < 1.0:
            pb.addUserDebugLine(pos_from, hit_position, ray_colors[i], lifeTime=0.1)
        else:
            pb.addUserDebugLine(pos_from, ray_end, [0.5, 0.5, 0.5], lifeTime=0.1)
    
    dist_0, dist_45, dist_90, dist_135 = ray_distances

    speeds = calculate_wheel_speeds(dist_0, dist_45, dist_90, dist_135)
    
    pb.setJointMotorControlArray(
        bodyIndex=obj,
        jointIndices=motorIdx,
        targetVelocities=speeds,
        controlMode=pb.VELOCITY_CONTROL,
    )
    
    print(f"Время: {t:.1f}с, Позиция: ({pos[0]:.2f}, {pos[1]:.2f}), "
            f"Расстояния: {dist_0:.2f}, {dist_45:.2f}, {dist_90:.2f}, {dist_135:.2f}, "
            f"Скорости: {speeds[0]:.1f}, {speeds[1]:.1f}")
    
    pb.stepSimulation()