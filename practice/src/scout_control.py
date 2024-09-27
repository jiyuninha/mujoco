import mujoco_py
import os

model_path = '/home/lee/simulation/practice/assets/env/scout_env.xml'

# 모델 파일 존재 여부 확인
if not os.path.exists(model_path):
    print(f"파일이 존재하지 않습니다: {model_path}")
else:
    try:
        model = mujoco_py.load_model_from_path(model_path)
        print("모델이 성공적으로 로드되었습니다")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        model = None

    if model is not None:
        try:
            sim = mujoco_py.MjSim(model)
            viewer = mujoco_py.MjViewer(sim)
            print("시뮬레이션이 성공적으로 초기화되었습니다")
        except Exception as e:
            print(f"시뮬레이션 초기화 중 오류 발생: {e}")
            sim = None
            viewer = None

        if sim is not None and viewer is not None:
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
                    # actuator의 ctrl 값을 설정
                    wheel_speeds_update = {
                        'front_right_wheel_motor': 0.5,
                        'front_left_wheel_motor': 0.5,
                        'rear_right_wheel_motor': 0.5,
                        'rear_left_wheel_motor': 0.5
                    }

                    # 각 모터에 제어 신호 전달
                    for actuator_name, ctrl_value in wheel_speeds_update.items():
                        actuator_id = model.actuator_name2id(actuator_name)
                        
                        sim.data.ctrl[actuator_id] = ctrl_value

                    # 시뮬레이션 스텝 진행
                    sim.step()

                    # 현재 위치와 속도 출력
                    wheel_positions = {
                        'front_right_wheel_joint': sim.data.qpos[model.joint_name2id('front_right_wheel_joint')],
                        'front_left_wheel_joint': sim.data.qpos[model.joint_name2id('front_left_wheel_joint')],
                        'rear_right_wheel_joint': sim.data.qpos[model.joint_name2id('rear_right_wheel_joint')],
                        'rear_left_wheel_joint': sim.data.qpos[model.joint_name2id('rear_left_wheel_joint')]
                    }

                    wheel_velocities = {
                        'front_right_wheel_joint': sim.data.qvel[model.joint_name2id('front_right_wheel_joint')],
                        'front_left_wheel_joint': sim.data.qvel[model.joint_name2id('front_left_wheel_joint')],
                        'rear_right_wheel_joint': sim.data.qvel[model.joint_name2id('rear_right_wheel_joint')],
                        'rear_left_wheel_joint': sim.data.qvel[model.joint_name2id('rear_left_wheel_joint')]
                    }
                
                except Exception as e:
                    print(f"제어 설정 중 오류 발생: {e}")
                    break

                
                # 시뮬레이션 렌더링
                viewer.render()

            print("종료하려면 'q'를 누르십시오.")
            while True:
                viewer.render()
                if viewer.exit:
                    break
