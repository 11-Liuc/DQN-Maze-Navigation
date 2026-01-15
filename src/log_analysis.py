"""
训练日志分析模块
绘制训练过程的奖励曲线和分析报告
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform
from pathlib import Path


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


# 初始化中文字体
setup_chinese_font()


def plot_reward_curve(csv_path="training_logs.csv", save_path=None, show=True):
    """
    绘制训练奖励曲线

    Args:
        csv_path: CSV日志文件路径
        save_path: 图片保存路径（可选）
        show: 是否显示图片
    """
    data = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DQN 迷宫寻路训练分析', fontsize=14, fontweight='bold')

    # 1. 每回合奖励 + 100回合滑动平均
    ax1 = axes[0, 0]
    ax1.plot(data["episode"], data["total_reward"], alpha=0.3, label='每回合奖励')
    data["avg_reward_100"] = data["total_reward"].rolling(window=100, min_periods=1).mean()
    ax1.plot(data["episode"], data["avg_reward_100"], linewidth=2, label='100回合平均')
    ax1.set_xlabel('回合 (Episode)')
    ax1.set_ylabel('奖励 (Reward)')
    ax1.set_title('训练奖励曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Epsilon 衰减曲线
    ax2 = axes[0, 1]
    ax2.plot(data["episode"], data["epsilon"], color='orange', linewidth=2)
    ax2.set_xlabel('回合 (Episode)')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('探索率 (ε) 衰减曲线')
    ax2.grid(True, alpha=0.3)

    # 3. 每回合步数
    ax3 = axes[1, 0]
    ax3.plot(data["episode"], data["steps"], alpha=0.3, label='每回合步数')
    data["avg_steps_100"] = data["steps"].rolling(window=100, min_periods=1).mean()
    ax3.plot(data["episode"], data["avg_steps_100"], linewidth=2, color='green', label='100回合平均')
    ax3.set_xlabel('回合 (Episode)')
    ax3.set_ylabel('步数 (Steps)')
    ax3.set_title('每回合步数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 成功率（基于奖励判断）
    ax4 = axes[1, 1]
    # 奖励 > 0 通常意味着成功到达终点
    data["success"] = (data["total_reward"] > 0).astype(int)
    data["success_rate_100"] = data["success"].rolling(window=100, min_periods=1).mean() * 100
    ax4.plot(data["episode"], data["success_rate_100"], color='purple', linewidth=2)
    ax4.set_xlabel('回合 (Episode)')
    ax4.set_ylabel('成功率 (%)')
    ax4.set_title('100回合滑动成功率')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    if show:
        plt.show()

    return fig


def generate_training_report(csv_path="training_logs.csv"):
    """
    生成训练报告摘要

    Args:
        csv_path: CSV日志文件路径

    Returns:
        report: 报告字典
    """
    data = pd.read_csv(csv_path)

    # 计算统计指标
    total_episodes = len(data)
    final_reward = data.iloc[-1]["total_reward"]
    best_reward = data["total_reward"].max()
    worst_reward = data["total_reward"].min()

    # 计算收敛点（首次连续10回合平均奖励 > 5）
    data["avg_10"] = data["total_reward"].rolling(window=10).mean()
    convergence_episode = None
    for i, avg in enumerate(data["avg_10"]):
        if avg is not None and avg > 5:
            convergence_episode = i
            break

    # 最后100回合统计
    last_100 = data.tail(100)
    success_count = (last_100["total_reward"] > 0).sum()
    avg_steps_last_100 = last_100["steps"].mean()

    report = {
        "总训练回合": total_episodes,
        "最终奖励": round(final_reward, 2),
        "最佳奖励": round(best_reward, 2),
        "最差奖励": round(worst_reward, 2),
        "收敛回合": convergence_episode,
        "最后100回合成功率": f"{success_count}%",
        "最后100回合平均步数": round(avg_steps_last_100, 1)
    }

    print("\n" + "=" * 50)
    print("训练报告摘要")
    print("=" * 50)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("=" * 50)

    return report


if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "training_logs.csv"

    if not Path(csv_file).exists():
        print(f"错误: 找不到日志文件 {csv_file}")
        print("请先运行训练: python train_manager.py")
        sys.exit(1)

    # 绘制曲线
    plot_reward_curve(csv_file, save_path="training_curves.png")

    # 生成报告
    generate_training_report(csv_file)
