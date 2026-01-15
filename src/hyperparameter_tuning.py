"""
超参数调优模块
对比不同超参数配置对训练效果的影响
"""

import sys
import os
import csv
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import platform
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from maze_env import MazeEnv
from dqn_agent import DQNAgent


def setup_chinese_font():
    """配置中文字体支持"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['axes.unicode_minus'] = False


setup_chinese_font()


def train_with_config(config, episodes=300, verbose=False):
    """
    使用指定配置训练智能体

    Args:
        config: 超参数配置字典
        episodes: 训练回合数
        verbose: 是否输出训练过程

    Returns:
        history: 训练历史记录
    """
    env = MazeEnv(maze_id=0)
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        lr=config.get('lr', 0.001),
        gamma=config.get('gamma', 0.99),
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=config.get('epsilon_decay', 0.99)
    )

    history = {
        'episode': [],
        'reward': [],
        'epsilon': [],
        'steps': []
    }

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(200):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        history['episode'].append(episode)
        history['reward'].append(total_reward)
        history['epsilon'].append(agent.epsilon)
        history['steps'].append(steps)

        if verbose and episode % 50 == 0:
            print(f"  Episode {episode}: Reward={total_reward:.2f}, Epsilon={agent.epsilon:.3f}")

    return history


def compare_learning_rates():
    """对比不同学习率的效果"""
    print("\n" + "=" * 60)
    print("实验1: 学习率 (Learning Rate) 对比")
    print("=" * 60)

    lr_configs = [
        {'lr': 0.0001, 'label': 'lr=0.0001'},
        {'lr': 0.001, 'label': 'lr=0.001 (默认)'},
        {'lr': 0.01, 'label': 'lr=0.01'},
    ]

    results = {}
    for config in lr_configs:
        print(f"\n训练配置: {config['label']}")
        history = train_with_config(config, episodes=300, verbose=True)
        results[config['label']] = history

    return results, 'learning_rate'


def compare_gamma():
    """对比不同折扣因子的效果"""
    print("\n" + "=" * 60)
    print("实验2: 折扣因子 (Gamma) 对比")
    print("=" * 60)

    gamma_configs = [
        {'gamma': 0.9, 'label': 'γ=0.9'},
        {'gamma': 0.95, 'label': 'γ=0.95'},
        {'gamma': 0.99, 'label': 'γ=0.99 (默认)'},
    ]

    results = {}
    for config in gamma_configs:
        print(f"\n训练配置: {config['label']}")
        history = train_with_config(config, episodes=300, verbose=True)
        results[config['label']] = history

    return results, 'gamma'


def compare_epsilon_decay():
    """对比不同ε衰减策略的效果"""
    print("\n" + "=" * 60)
    print("实验3: ε衰减策略 (Epsilon Decay) 对比")
    print("=" * 60)

    decay_configs = [
        {'epsilon_decay': 0.95, 'label': 'decay=0.95 (快速衰减)'},
        {'epsilon_decay': 0.99, 'label': 'decay=0.99 (默认)'},
        {'epsilon_decay': 0.995, 'label': 'decay=0.995 (慢速衰减)'},
    ]

    results = {}
    for config in decay_configs:
        print(f"\n训练配置: {config['label']}")
        history = train_with_config(config, episodes=300, verbose=True)
        results[config['label']] = history

    return results, 'epsilon_decay'


def plot_comparison(results, experiment_name, save_path=None):
    """
    绘制对比图

    Args:
        results: 实验结果字典
        experiment_name: 实验名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 奖励曲线对比
    ax1 = axes[0]
    for label, history in results.items():
        rewards = pd.Series(history['reward']).rolling(window=50, min_periods=1).mean()
        ax1.plot(history['episode'], rewards, label=label, linewidth=2)

    ax1.set_xlabel('回合 (Episode)')
    ax1.set_ylabel('平均奖励 (50回合滑动)')
    ax1.set_title(f'{experiment_name} - 奖励曲线对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 最终性能对比（柱状图）
    ax2 = axes[1]
    labels = list(results.keys())
    final_rewards = [np.mean(results[l]['reward'][-50:]) for l in labels]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    bars = ax2.bar(range(len(labels)), final_rewards, color=colors)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('最后50回合平均奖励')
    ax2.set_title(f'{experiment_name} - 最终性能对比')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars, final_rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()


def generate_tuning_report(all_results):
    """
    生成超参数调优报告

    Args:
        all_results: 所有实验结果
    """
    report_path = Path(__file__).parent.parent / "outputs" / "hyperparameter_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 超参数调优报告\n\n")
        f.write("## 实验概述\n\n")
        f.write("本报告对比了不同超参数配置对 DQN 迷宫寻路智能体训练效果的影响。\n\n")

        for exp_name, (results, _) in all_results.items():
            f.write(f"## {exp_name}\n\n")
            f.write("| 配置 | 最后50回合平均奖励 | 最后50回合平均步数 | 收敛速度 |\n")
            f.write("|------|-------------------|-------------------|----------|\n")

            for label, history in results.items():
                avg_reward = np.mean(history['reward'][-50:])
                avg_steps = np.mean(history['steps'][-50:])

                # 计算收敛速度（首次达到奖励>5的回合）
                convergence = None
                rolling = pd.Series(history['reward']).rolling(window=10).mean()
                for i, r in enumerate(rolling):
                    if r is not None and r > 5:
                        convergence = i
                        break

                conv_str = str(convergence) if convergence else "未收敛"
                f.write(f"| {label} | {avg_reward:.2f} | {avg_steps:.1f} | {conv_str} |\n")

            f.write("\n")

        f.write("## 结论与建议\n\n")
        f.write("1. **学习率**: 0.001 是较为平衡的选择，过大会导致不稳定，过小收敛慢。\n")
        f.write("2. **折扣因子 γ**: 0.99 适合迷宫任务，能够更好地考虑长期奖励。\n")
        f.write("3. **ε衰减策略**: 0.99 的衰减率在探索与利用之间取得较好平衡。\n")

    print(f"\n报告已生成: {report_path}")


def run_all_experiments():
    """运行所有超参数调优实验"""
    print("=" * 60)
    print("DQN 超参数调优实验")
    print("=" * 60)

    # 确保 outputs 目录存在
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    all_results = {}

    # 实验1: 学习率对比
    results_lr, name_lr = compare_learning_rates()
    all_results['学习率对比'] = (results_lr, name_lr)
    plot_comparison(results_lr, '学习率', save_path='outputs/tuning_learning_rate.png')

    # 实验2: 折扣因子对比
    results_gamma, name_gamma = compare_gamma()
    all_results['折扣因子对比'] = (results_gamma, name_gamma)
    plot_comparison(results_gamma, '折扣因子 γ', save_path='outputs/tuning_gamma.png')

    # 实验3: ε衰减对比
    results_eps, name_eps = compare_epsilon_decay()
    all_results['ε衰减策略对比'] = (results_eps, name_eps)
    plot_comparison(results_eps, 'ε衰减策略', save_path='outputs/tuning_epsilon_decay.png')

    # 生成报告
    generate_tuning_report(all_results)

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
