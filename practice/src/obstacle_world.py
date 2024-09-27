import mujoco_py
import os

env_path = os.path.join('/home/lee/simulation/practice/assets/env/obstacle.xml')
env = mujoco_py.load_model_from_path(env_path)
sim = mujoco_py.MjSim(env)
viewer = mujoco_py.MjViewer(sim)

while True:
    viewer.render()