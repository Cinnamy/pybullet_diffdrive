import pybullet as pb
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

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

RAY_LENGTH = 10.0
SPEED = 30.0
DISTANCE_45 = 2.3
DISTANCE_0 = 1.5
DISTANCE_90 = 1

def calculate_wheel_speeds(dist_0, dist_45, dist_90, dist_135):
    d = 10.0

    if dist_0 < DISTANCE_0 or dist_45 < DISTANCE_45:
        return [SPEED, d]

    if dist_90 == RAY_LENGTH or dist_45 == RAY_LENGTH or dist_90 > DISTANCE_90:
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

maxTime = 80 
dt = 1 / 60
logTime = np.arange(0, maxTime, dt)

angles_local = [np.pi/2, np.pi/4, 0, -np.pi/4]  # 0°, 45°, 90°, 135°

ray_colors = [
    [1, 0, 0],  # red 0°
    [0, 1, 0],  # green 45°
    [0, 0, 1],  # blue 90°
    [1, 1, 0]   # yellow 135°
]

robot_positions = []
lidar_points = []

for t in logTime:

    pos, orn = pb.getBasePositionAndOrientation(obj)
    euler = pb.getEulerFromQuaternion(orn)
    angle = euler[2]

    robot_positions.append([pos[0], pos[1]])
    
    # measure the distance to obstacles with rays

    ray_angles = [angle + np.pi/2, angle + np.pi/4, angle, angle - np.pi/4]

    pos_from = np.tile([pos[0], pos[1], pos[2] + 0.1], (4,1))

    ray_end_x = pos[0] + RAY_LENGTH * np.cos(ray_angles)
    ray_end_y = pos[1] + RAY_LENGTH * np.sin(ray_angles)
    ray_end_z = np.full(4, pos[2] + 0.1)
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
            pb.addUserDebugLine(pos_from[i], hit_positions[i], ray_colors[i], lifeTime=0.1)
            lidar_points.append([hit_positions[i][0], hit_positions[i][1]])
        else:
            pb.addUserDebugLine(pos_from[i], pos_to[i], [0.5, 0.5, 0.5], lifeTime=0.1)
    
    dist_0, dist_45, dist_90, dist_135 = distances

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

robot_positions = np.array(robot_positions)
lidar_points = np.array(lidar_points)

plt.figure(figsize=(10, 8))
plt.scatter(lidar_points[:, 0], lidar_points[:, 1], 
            c='red', s=5, alpha=0.3, label='walls')
plt.plot(robot_positions[:, 0], robot_positions[:, 1], 
         'b-', linewidth=2, label='robot\'s path')
plt.scatter(robot_positions[0, 0], robot_positions[0, 1], 
            c='green', s=100, marker='o', label='start')
plt.scatter(robot_positions[-1, 0], robot_positions[-1, 1], 
            c='orange', s=100, marker='o', label='end')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot\'s path and walls')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()