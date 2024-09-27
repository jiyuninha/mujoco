import numpy as np
from stable_baselines3 import PPO
from env import ScoutEnv

# 목표 지점 설정 (예: [1.0, 1.0])
target_point = np.array([1.0, 1.0])

# 환경 생성
env = ScoutEnv(model_path='/home/lee/simulation/practice/assets/env/scout_env.xml', target_point=target_point)

# 모델 초기화
model = PPO("MlpPolicy", env, verbose=1)

# 초기 위치와 목표 라인 설정
# initial_position = np.array([3.0, 3.0])  # 스카우트의 초기 위치
goal_x, goal_y = 0.0, 0.0  # 목표 라인의 y 좌표
best_model_path = "best_scout_model.zip"
# best_trajectory_score = float('inf')  # 초기값을 무한대로 설정

# 최대 스텝 수 설정
max_steps = 5000  # 각 에피소드에서 최대 5000 스텝까지만 허용

# 학습
total_timesteps = 1000
for timestep in range(0, total_timesteps, 100000):  # 1000 timesteps마다 학습 진행
    print(f"timestep: {timestep}\n")
    obs = env.reset()
    done = False
    trajectory_score = 0  # 각 에피소드의 궤적 점수를 저장
    step_count = 0

    while not done and step_count < max_steps:  # 최대 스텝 수를 넘지 않도록 제한
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # 현재 위치
        scout_position = obs[:2]
        
        # 목표 직선 경로에서의 오차 계산 (y = 0 직선 경로와의 거리)
        ideal_position = np.array([scout_position[0], goal_y])
        trajectory_score += np.linalg.norm(scout_position - ideal_position)

        # 현재 상태를 렌더링 (선택사항: 학습 중 렌더링)
        env.render()

        step_count += 1  # 스텝 카운트 증가

    # 모델 학습 진행 (1000 timesteps마다 학습)
    model.learn(total_timesteps=1000, reset_num_timesteps=False)

    # 현재 모델이 가장 좋은 궤적 점수를 가졌는지 확인
    if trajectory_score < best_trajectory_score:
        best_trajectory_score = trajectory_score
        model.save(best_model_path)
        print(f"New best model saved with trajectory score: {trajectory_score}")

print("Training completed.")
