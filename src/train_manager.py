import csv
from dqn_agent import DQNAgent
from maze_env import MazeEnv


def train():
    # ===== 1. 环境与智能体初始化 =====
    env = MazeEnv()
    agent = DQNAgent(
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    max_episodes = 500      # 训练回合数
    max_steps = 200         # 每回合最大步数

    # ===== 2. 创建 CSV 文件并写表头 =====
    with open("training_logs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "total_reward",
            "epsilon",
            "steps"
        ])

        # ===== 3. 训练循环 =====
        for episode in range(max_episodes):
            state = env.reset()
            total_reward = 0
            step_count = 0

            for step in range(max_steps):
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)

                agent.store_transition(
                    state, action, reward, next_state, done
                )
                agent.learn()

                state = next_state
                total_reward += reward
                step_count += 1

                if done:
                    break

            # 每回合结束后衰减 epsilon
            agent.decay_epsilon()

            # ===== 4. 写入 CSV 日志 =====
            writer.writerow([
                episode,
                round(total_reward, 2),
                round(agent.epsilon, 4),
                step_count
            ])

            # ===== 5. 终端输出（便于观察） =====
            if episode % 20 == 0:
                print(
                    f"Episode {episode:3d} | "
                    f"Reward: {total_reward:7.2f} | "
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Steps: {step_count}"
                )


if __name__ == "__main__":
    train()
