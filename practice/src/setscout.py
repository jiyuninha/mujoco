import mujoco_py

xml_path = '../assets/scout/scout.xml'
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)

print(sim.data.qpos)

while True:
    sim.step()
    viewer.render()