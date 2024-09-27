import mujoco_py
import os
import numpy as np

model_path = '/home/lee/simulation/practice/assets/env/scout_env.xml'
target_point = [5.0, 5.0]
waypoints = [[3.5,3.5],[4.0,4.0],[4.5,4.5]]

def drive_for_target_point():
    # 모델 파일 존재 여부 확인
    if not os.path.exists(model_path):
        print(f"파일이 존재하지 않습니다: {model_path}")
        return

    try:
        model = mujoco_py.load_model_from_path(model_path)
        print("모델이 성공적으로 로드되었습니다")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    try:
        sim = mujoco_py.MjSim(model)
        viewer = mujoco_py.MjViewer(sim)
        print("시뮬레이션이 성공적으로 초기화되었습니다")
    except Exception as e:
        print(f"시뮬레이션 초기화 중 오류 발생: {e}")
        return

    # 가용 액추에이터 출력
    actuator_names = [model.names[model.name_actuatoradr[i]] for i in range(model.nu)]
    print("사용 가능한 액추에이터:", actuator_names)

    for actuator_id in range(model.nu):
        actuator_name = model.names[model.name_actuatoradr[actuator_id]]
        joint_id = model.actuator_trnid[actuator_id][0]
        joint_name = model.names[model.name_jntadr[joint_id]]
        print(f"액추에이터 '{actuator_name}'는 조인트 '{joint_name}'에 연결되어 있습니다.")

    # 시뮬레이션 루프
    print("시뮬레이션 시작")
    while True:
        try:
            # 현재 스카우트의 위치 받아오기
            scout_position = sim.data.get_body_xpos("scout_base")
            scout_orientation = sim.data.get_body_xquat("scout_base")
            print(f"현재 위치: {scout_position}, 방향: {scout_orientation}")

            # 목표 지점으로의 벡터 계산
            direction_vector = np.array(target_point) - np.array([scout_position[0], scout_position[1]])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)  # 단위 벡터로 정규화

            # 현재 로봇의 진행 방향과 목표 방향 간의 각도 계산
            current_forward_vector = np.array([1, 0])  # 로봇의 기본 진행 방향 (x축 방향)
            rotation_matrix = np.array([
                [np.cos(scout_orientation[2]), -np.sin(scout_orientation[2])],
                [np.sin(scout_orientation[2]), np.cos(scout_orientation[2])]
            ])
            current_forward_vector = rotation_matrix.dot(current_forward_vector)

            angle_to_target = np.arctan2(direction_vector[1], direction_vector[0]) - np.arctan2(current_forward_vector[1], current_forward_vector[0])
            angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi] 범위로 정규화

            # 단순한 PID 컨트롤러를 사용하여 바퀴 속도 설정
            speed = 0.5  # 기본 속도
            turn_rate = angle_to_target * 2.0  # 회전 속도 비례 상수

            # 바퀴 속도 업데이트
            wheel_speeds_update = {
                'front_right_wheel_motor': speed - turn_rate,
                'front_left_wheel_motor': speed + turn_rate,
                'rear_right_wheel_motor': speed - turn_rate,
                'rear_left_wheel_motor': speed + turn_rate
            }

            # 각 모터에 제어 신호 전달
            for actuator_name, ctrl_value in wheel_speeds_update.items():
                actuator_id = model.actuator_name2id(actuator_name)
                sim.data.ctrl[actuator_id] = ctrl_value

            # 시뮬레이션 스텝 진행
            sim.step()
            viewer.render()

            # 목표 지점에 도달했는지 확인
            if np.linalg.norm(np.array(target_point) - np.array([scout_position[0], scout_position[1]])) < 0.1:
                print("목표 지점에 도달했습니다.")
                break

        except Exception as e:
            print(f"제어 설정 중 오류 발생: {e}")
            break

    print("종료하려면 'q'를 누르십시오.")
    while True:
        viewer.render()
        if viewer.exit:
            break

if __name__ == "__main__":
    drive_for_target_point()
