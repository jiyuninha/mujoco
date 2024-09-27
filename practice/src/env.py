import gym
from gym import spaces
import mujoco_py
import numpy as np
import os

class ScoutEnv(gym.Env):
    def __init__(self, model_path, target_point):
        super(ScoutEnv, self).__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {model_path}")

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)

        self.target_point = np.array(target_point)
        self.previous_distance = None  # 이전 위치에서 목표 지점까지의 거리

        # Action space: 바퀴 속도를 제어할 수 있도록 설정 (4개의 바퀴에 대한 속도)
        # 속도 범위를 -3.0에서 3.0으로 확장
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(4,), dtype=np.float32)

        # Observation space: 스카우트의 위치 및 속도만 포함
        obs_dim = 4  # x, y 위치 + x, y 속도 (총 4차원)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        try:
            # action은 각 바퀴의 속도를 나타냄
            # 각 action 값에 스케일링 팩터(예: 2.0)를 곱하여 제어 입력을 증폭
            scaling_factor = 2.0
            wheel_speeds_update = {
                'front_right_wheel_motor': action[0] * scaling_factor,
                'front_left_wheel_motor': action[1] * scaling_factor,
                'rear_right_wheel_motor': action[2] * scaling_factor,
                'rear_left_wheel_motor': action[3] * scaling_factor,
            }

            # 각 모터에 제어 신호 전달
            for actuator_name, ctrl_value in wheel_speeds_update.items():
                actuator_id = self.model.actuator_name2id(actuator_name)
                self.sim.data.ctrl[actuator_id] = ctrl_value

            # 시뮬레이션 스텝 진행
            self.sim.step()

            # 현재 위치 및 속도 계산
            scout_position = self.sim.data.get_body_xpos("scout_base")[:2]  # x, y 위치
            scout_velocity = self.sim.data.qvel.flat[:2]  # x, y 속도
            obs = np.concatenate([scout_position, scout_velocity])

            # 목표 지점까지의 거리 계산
            current_distance = np.linalg.norm(self.target_point - scout_position)

            # 목표 방향 벡터와 현재 속도 벡터의 코사인 유사도를 이용한 보상 계산
            direction_to_target = (self.target_point - scout_position) / (np.linalg.norm(self.target_point - scout_position) + 1e-8)
            velocity_direction = scout_velocity / (np.linalg.norm(scout_velocity) + 1e-8)  # 0으로 나누지 않도록 방지

            direction_similarity = np.dot(direction_to_target, velocity_direction)  # 두 벡터의 코사인 유사도
            direction_similarity = np.clip(direction_similarity, -1.0, 1.0)  # 유사도 범위를 -1에서 1 사이로 제한

            # 보상 계산
            reward = 0.0
            reward += 5.0 * direction_similarity  # 방향 보상: 목표 방향과 일치할수록 큰 보상
            reward -= 0.5 * (1.0 - direction_similarity)  # 방향이 틀릴수록 페널티

            if self.previous_distance is None:
                # 첫 번째 스텝에서는 previous_distance를 초기화만 하고, 보상 계산에 사용하지 않음
                self.previous_distance = current_distance
            else:
                # 거리 보상: 이전 스텝보다 목표 지점에 가까워졌다면 보상
                reward += 2.0 * (self.previous_distance - current_distance)
                self.previous_distance = current_distance

            if current_distance < 0.1:
                reward += 10.0  # 목표 지점 도달 보상
                return obs, reward, True, {}  # 목표 지점에 도달했으면 종료

            return obs, reward, False, {}

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"스텝 진행 중 오류 발생: {e}")
            return None, 0.0, True, {}
        
    def reset(self):
        # 초기 상태로 리셋
        self.sim.reset()
        scout_position = self.sim.data.get_body_xpos("scout_base")[:2]
        scout_velocity = self.sim.data.qvel.flat[:2]
        self.previous_distance = None  # 거리 초기화
        return np.concatenate([scout_position, scout_velocity])

    def render(self, mode='human'):
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
