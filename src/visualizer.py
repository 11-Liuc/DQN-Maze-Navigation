"""
可视化模块
实现迷宫环境和智能体轨迹的可视化展示
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import platform
from maze_env import MazeEnv


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


class MazeVisualizer:
    """迷宫可视化器"""

    def __init__(self, env: MazeEnv):
        """
        初始化可视化器

        Args:
            env: MazeEnv 实例
        """
        self.env = env
        self.fig = None
        self.ax = None

        # 颜色配置
        self.colors = {
            'wall': '#2C3E50',      # 深蓝灰色墙壁
            'path': '#ECF0F1',      # 浅灰色通路
            'start': '#27AE60',     # 绿色起点
            'goal': '#E74C3C',      # 红色终点
            'agent': '#3498DB',     # 蓝色智能体
            'trajectory': '#F39C12' # 橙色轨迹
        }

    def render_maze(self, show_grid=True):
        """
        渲染静态迷宫

        Args:
            show_grid: 是否显示网格线
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        maze = self.env.maze
        size = self.env.maze_size

        # 绘制迷宫格子
        for i in range(size):
            for j in range(size):
                color = self.colors['wall'] if maze[i, j] == 1 else self.colors['path']
                rect = patches.Rectangle(
                    (j, size - 1 - i), 1, 1,
                    linewidth=1,
                    edgecolor='gray' if show_grid else color,
                    facecolor=color
                )
                self.ax.add_patch(rect)

        # 标记起点
        start_rect = patches.Rectangle(
            (self.env.start[1], size - 1 - self.env.start[0]), 1, 1,
            linewidth=2,
            edgecolor='black',
            facecolor=self.colors['start'],
            alpha=0.7
        )
        self.ax.add_patch(start_rect)
        self.ax.text(
            self.env.start[1] + 0.5, size - 0.5 - self.env.start[0],
            'S', ha='center', va='center', fontsize=16, fontweight='bold'
        )

        # 标记终点
        goal_rect = patches.Rectangle(
            (self.env.goal[1], size - 1 - self.env.goal[0]), 1, 1,
            linewidth=2,
            edgecolor='black',
            facecolor=self.colors['goal'],
            alpha=0.7
        )
        self.ax.add_patch(goal_rect)
        self.ax.text(
            self.env.goal[1] + 0.5, size - 0.5 - self.env.goal[0],
            'G', ha='center', va='center', fontsize=16, fontweight='bold'
        )

        self.ax.set_xlim(0, size)
        self.ax.set_ylim(0, size)
        self.ax.set_aspect('equal')
        self.ax.set_title('DQN 迷宫环境', fontsize=14)

        return self.fig, self.ax

    def plot_trajectory(self, trajectory, title="智能体移动轨迹"):
        """
        绘制智能体移动轨迹

        Args:
            trajectory: 状态列表 [(x1,y1), (x2,y2), ...]
            title: 图表标题
        """
        self.render_maze()
        size = self.env.maze_size

        if len(trajectory) < 2:
            plt.title(title)
            plt.show()
            return

        # 转换坐标并绘制轨迹线
        xs = [pos[1] + 0.5 for pos in trajectory]
        ys = [size - 0.5 - pos[0] for pos in trajectory]

        # 绘制轨迹线
        self.ax.plot(
            xs, ys,
            color=self.colors['trajectory'],
            linewidth=2,
            marker='o',
            markersize=6,
            alpha=0.8,
            label='轨迹'
        )

        # 标记起始位置和结束位置
        self.ax.plot(xs[0], ys[0], 'go', markersize=12, label='起始')
        self.ax.plot(xs[-1], ys[-1], 'r*', markersize=15, label='结束')

        # 添加步数标注
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i % max(1, len(trajectory) // 10) == 0:  # 每隔几步标注
                self.ax.annotate(
                    str(i), (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8
                )

        self.ax.legend(loc='upper right')
        self.ax.set_title(f'{title} (共 {len(trajectory)-1} 步)', fontsize=14)
        plt.show()

    def animate_trajectory(self, trajectory, interval=300, save_path=None):
        """
        动画展示智能体移动轨迹

        Args:
            trajectory: 状态列表
            interval: 帧间隔(毫秒)
            save_path: 保存路径 (可选，如 'demo.gif')
        """
        self.render_maze()
        size = self.env.maze_size

        # 创建智能体圆形
        agent_circle = patches.Circle(
            (trajectory[0][1] + 0.5, size - 0.5 - trajectory[0][0]),
            0.3,
            color=self.colors['agent'],
            zorder=10
        )
        self.ax.add_patch(agent_circle)

        # 轨迹线
        line, = self.ax.plot([], [], color=self.colors['trajectory'],
                             linewidth=2, alpha=0.6)

        # 步数文本
        step_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        xs, ys = [], []

        def init():
            line.set_data([], [])
            step_text.set_text('')
            return agent_circle, line, step_text

        def animate(frame):
            pos = trajectory[frame]
            x = pos[1] + 0.5
            y = size - 0.5 - pos[0]

            # 更新智能体位置
            agent_circle.center = (x, y)

            # 更新轨迹
            xs.append(x)
            ys.append(y)
            line.set_data(xs, ys)

            # 更新步数
            step_text.set_text(f'步数: {frame}')

            return agent_circle, line, step_text

        anim = animation.FuncAnimation(
            self.fig, animate, init_func=init,
            frames=len(trajectory), interval=interval,
            blit=True, repeat=False
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"动画已保存至: {save_path}")

        plt.show()
        return anim


def visualize_trained_agent(agent, env, max_steps=50):
    """
    可视化训练好的智能体

    Args:
        agent: 训练好的 DQNAgent
        env: MazeEnv 环境
        max_steps: 最大步数

    Returns:
        trajectory: 轨迹列表
    """
    state = env.reset()
    trajectory = [tuple(state)]

    for _ in range(max_steps):
        # 使用贪婪策略（不探索）
        old_epsilon = agent.epsilon
        agent.epsilon = 0
        action = agent.choose_action(state)
        agent.epsilon = old_epsilon

        next_state, reward, done = env.step(action)
        trajectory.append(tuple(next_state))
        state = next_state

        if done:
            break

    return trajectory


if __name__ == "__main__":
    # 测试可视化
    env = MazeEnv(maze_id=0)
    viz = MazeVisualizer(env)

    # 1. 渲染静态迷宫
    print("渲染静态迷宫...")
    viz.render_maze()
    plt.show()

    # 2. 模拟一条随机轨迹并可视化
    print("\n生成随机轨迹...")
    import random
    env.reset()
    trajectory = [tuple(env.state)]
    for _ in range(20):
        action = random.randint(0, 3)
        next_state, reward, done = env.step(action)
        trajectory.append(tuple(next_state))
        if done:
            break

    print(f"轨迹长度: {len(trajectory)}")
    viz.plot_trajectory(trajectory, "随机移动轨迹示例")
