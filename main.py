"""
DQN 迷宫寻路智能体 - 主入口
课题5：基于深度Q网络的迷宫寻路智能体

使用方法:
    python main.py --mode train       # 训练模式
    python main.py --mode test        # 测试模式
    python main.py --mode visualize   # 可视化模式
    python main.py --mode plot        # 绘制训练曲线
    python main.py --mode tuning      # 超参数调优
"""

import argparse
import sys
import os
import platform

# 将 src 目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maze_env import MazeEnv
from dqn_agent import DQNAgent
from visualizer import MazeVisualizer, visualize_trained_agent
import torch


def setup_chinese_font():
    """配置中文字体支持"""
    import matplotlib.pyplot as plt
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['axes.unicode_minus'] = False


def train(args):
    """训练模式"""
    import csv

    print("=" * 50)
    print("DQN 迷宫寻路智能体 - 训练模式")
    print("=" * 50)

    # 确保 outputs 目录存在
    os.makedirs("outputs", exist_ok=True)

    # 初始化环境和智能体
    env = MazeEnv(maze_id=args.maze_id)
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=args.epsilon_decay
    )

    print(f"\n迷宫配置: maze_id={args.maze_id}")
    print(f"超参数: lr={args.lr}, gamma={args.gamma}, epsilon_decay={args.epsilon_decay}")
    print(f"训练回合: {args.episodes}, 每回合最大步数: {args.max_steps}\n")

    # 创建 CSV 日志文件
    csv_file = open("outputs/training_logs.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "total_reward", "epsilon", "steps"])

    # 训练循环
    best_reward = float('-inf')
    recent_rewards = []  # 记录最近的奖励用于判断稳定性
    best_avg_reward = float('-inf')

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0

        for step in range(args.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break

        # 每回合结束后衰减 epsilon
        agent.decay_epsilon()

        # 写入 CSV 日志
        csv_writer.writerow([episode, round(total_reward, 2), round(agent.epsilon, 4), step_count])

        # 记录最近50回合奖励
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)

        # 更新最佳奖励
        if total_reward > best_reward:
            best_reward = total_reward

        # 基于最近50回合平均奖励保存模型（更稳定）
        if len(recent_rewards) >= 50:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy_net.state_dict(), 'outputs/best_model.pth')

        # 定期输出
        if episode % 20 == 0:
            avg_50 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | "
                  f"Avg50: {avg_50:.2f} | Epsilon: {agent.epsilon:.3f}")

    csv_file.close()

    # 保存最终模型
    torch.save(agent.policy_net.state_dict(), 'outputs/final_model.pth')
    print("\n" + "=" * 50)
    print(f"训练完成! 最佳奖励: {best_reward:.2f}")
    print("模型已保存: best_model.pth, final_model.pth")
    print("=" * 50)

    return agent


def test(args):
    """测试模式"""
    print("=" * 50)
    print("DQN 迷宫寻路智能体 - 测试模式")
    print("=" * 50)

    env = MazeEnv(maze_id=args.maze_id)
    agent = DQNAgent(state_dim=2, action_dim=4)

    # 加载模型
    model_path = args.model if args.model else 'outputs/best_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在，请先训练模型")
        return

    agent.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
    agent.epsilon = 0  # 测试时不探索
    print(f"已加载模型: {model_path}\n")

    # 运行测试
    success_count = 0
    total_steps = 0

    for episode in range(args.test_episodes):
        state = env.reset()
        steps = 0

        for step in range(args.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1

            if done:
                if tuple(state) == env.goal:
                    success_count += 1
                break

        total_steps += steps
        print(f"测试 {episode+1}: 步数={steps}, 到达终点={'是' if tuple(state)==env.goal else '否'}")

    print("\n" + "=" * 50)
    print(f"测试结果: 成功率 {success_count}/{args.test_episodes} "
          f"({100*success_count/args.test_episodes:.1f}%)")
    print(f"平均步数: {total_steps/args.test_episodes:.1f}")
    print("=" * 50)


def visualize(args):
    """可视化模式"""
    print("=" * 50)
    print("DQN 迷宫寻路智能体 - 可视化模式")
    print("=" * 50)

    env = MazeEnv(maze_id=args.maze_id)
    viz = MazeVisualizer(env)

    if args.model and os.path.exists(args.model):
        # 可视化训练好的智能体
        agent = DQNAgent(state_dim=2, action_dim=4)
        agent.policy_net.load_state_dict(torch.load(args.model, weights_only=True))
        print(f"已加载模型: {args.model}")

        trajectory = visualize_trained_agent(agent, env, max_steps=args.max_steps)
        print(f"轨迹长度: {len(trajectory)} 步")

        if args.animate:
            viz.animate_trajectory(trajectory, interval=300, save_path=args.save_gif)
        else:
            viz.plot_trajectory(trajectory, "训练智能体轨迹")
    else:
        # 仅显示迷宫
        print("未加载模型，仅显示迷宫结构")
        viz.render_maze()
        import matplotlib.pyplot as plt
        plt.show()


def plot_curves(args):
    """绘制训练曲线"""
    setup_chinese_font()
    from log_analysis import plot_reward_curve, generate_training_report

    csv_path = args.csv if args.csv else "outputs/training_logs.csv"
    if not os.path.exists(csv_path):
        print(f"错误: 找不到日志文件 {csv_path}")
        print("请先运行训练: python main.py --mode train")
        return

    print("=" * 50)
    print("DQN 迷宫寻路智能体 - 训练曲线分析")
    print("=" * 50)

    # 绘制曲线
    plot_reward_curve(csv_path, save_path="outputs/training_curves.png")

    # 生成报告
    generate_training_report(csv_path)


def tuning(args):
    """超参数调优"""
    setup_chinese_font()
    from hyperparameter_tuning import run_all_experiments

    print("=" * 50)
    print("DQN 迷宫寻路智能体 - 超参数调优")
    print("=" * 50)
    print("注意: 此过程需要较长时间，请耐心等待...")

    run_all_experiments()


def main():
    parser = argparse.ArgumentParser(
        description='DQN 迷宫寻路智能体',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'visualize', 'plot', 'tuning'],
                        help='运行模式: train/test/visualize/plot/tuning')

    # 环境参数
    parser.add_argument('--maze_id', type=int, default=0,
                        help='迷宫ID (0-3)')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='每回合最大步数')

    # 训练参数
    parser.add_argument('--episodes', type=int, default=500,
                        help='训练回合数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Epsilon衰减率')

    # 测试参数
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='测试回合数')
    parser.add_argument('--model', type=str, default=None,
                        help='模型文件路径')

    # 可视化参数
    parser.add_argument('--animate', action='store_true',
                        help='使用动画展示')
    parser.add_argument('--save_gif', type=str, default=None,
                        help='保存GIF路径')

    # 分析参数
    parser.add_argument('--csv', type=str, default=None,
                        help='训练日志CSV文件路径')

    args = parser.parse_args()

    # 根据模式执行
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'visualize':
        setup_chinese_font()
        visualize(args)
    elif args.mode == 'plot':
        plot_curves(args)
    elif args.mode == 'tuning':
        tuning(args)


if __name__ == '__main__':
    main()
