from stable_baselines3 import DQN
from env_dqn import DiscreteScoutEnv

# 환경 생성
env = DiscreteScoutEnv(model_path='/home/lee/simulation/practice/assets/env/scout_env.xml', 
               target_point=[5.0, 5.0], 
               waypoints=[[3.5,3.5],[3.75,3.75],[4.0,4.0],[4.25,4.25],[4.5,4.5],[4.75,4.75],[5.0,5.0]])

# 모델 초기화 (TensorBoard 로그 경로 설정)
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="../train_log/dqn_scout_tensorboard/")

# 모델 학습
model.learn(total_timesteps=10000000, tb_log_name="DQN_scout_run")

# 모델 저장
model.save("dqn_scout_model")

# 학습된 모델로 시뮬레이션 실행
obs = env.reset()
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
