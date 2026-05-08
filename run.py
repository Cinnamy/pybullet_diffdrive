import pybullet as pb
import pybullet_data
import numpy as np
import time

pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
# load the plane to stand onto
pb.loadURDF("plane.urdf")
# elevate robot so that the wheels are touching the plane
obj = pb.loadURDF("diff_drive.urdf.xml", 0, 0, 0.1)
wallsId = pb.loadURDF("walls.urdf.xml", useFixedBase=True)

pb.setGravity(0, 0, -9.8)
# wheel order: [right, left]
motorIdx = [0, 1]
# const wheel velocities for test
vel = [6.0, 3.0]
maxTime = 5  # seconds
dt = 1 / 240  # seconds
logTime = np.arange(0, maxTime, dt)
sz = len(logTime)
logX = np.zeros(sz)
logY = np.zeros(sz)

# turn off castor wheel motor for free motion
pb.setJointMotorControl2(
    bodyIndex=obj,
    jointIndex=2,
    targetVelocity=0,
    controlMode=pb.VELOCITY_CONTROL,
    force=0,
)

# apply some wheel velocity
pb.setJointMotorControlArray(
    bodyIndex=obj,
    jointIndices=motorIdx,
    targetVelocities=vel,
    controlMode=pb.VELOCITY_CONTROL,
)

rayLength = 1.0
angles = np.linspace(0.0, 2*np.pi, 360)
x = rayLength * np.cos(angles)
y = rayLength * np.sin(angles)
h = 0.5
z = h*np.ones(360)
posFrom = np.tile([0,0,h], (360,1))
posTo = np.vstack((x,y,z)).T
print(posFrom)
print(posTo)
res = pb.rayTestBatch(
    posFrom,
    posTo
)
coords = [arr[3] for arr in res]
xy = np.array([[coord[0], coord[1]] for coord in coords])
print(xy)

idx = 0
for t in logTime:
    pb.stepSimulation()
    posAndOrn = pb.getBasePositionAndOrientation(obj)
    pos = posAndOrn[0]
    logX[idx] = pos[0]
    logY[idx] = pos[1]
    idx += 1
    # time.sleep(dt)

pb.disconnect()

# show obstacles XY plot
import matplotlib.pyplot as plt

plt.plot(xy[:,0], xy[:,1])
plt.axis("equal")
plt.show()
