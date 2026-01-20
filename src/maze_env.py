"""
迷宫环境模块
实现 5x5 迷宫，兼容 Gymnasium 接口风格
"""

import json
import numpy as np
from pathlib import Path

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class MazeEnv:
    """
    5x5 迷宫环境（兼容 Gymnasium 接口）

    状态空间: (x, y) 坐标，范围 [0, 4]
    动作空间: 0=上, 1=下, 2=左, 3=右

    奖励设计:
        - 到达终点: +10
        - 碰撞墙壁: -5
        - 每步惩罚: -0.1
    """

    # 动作映射: 上、下、左、右
    ACTIONS = {
        0: (-1, 0),  # 上
        1: (1, 0),   # 下
        2: (0, -1),  # 左
        3: (0, 1)    # 右
    }
    ACTION_NAMES = ['上', '下', '左', '右']

    def __init__(self, maze_id=0, random_obstacles=False, obstacle_ratio=0.2, maze_size=5):
        """
        初始化迷宫环境

        Args:
            maze_id: 迷宫配置编号，默认为0
            random_obstacles: 是否随机生成障碍物
            obstacle_ratio: 随机障碍物比例（仅当 random_obstacles=True 时生效）
            maze_size: 迷宫大小，默认为5
        """
        self.maze_size = maze_size
        self.state_dim = 10
        self.action_dim = 4
        self.random_obstacles = random_obstacles
        self.obstacle_ratio = obstacle_ratio

        # Gymnasium 兼容的空间定义（归一化坐标 + 8方向障碍物检测）
        if HAS_GYM:
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(10,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(4)

        # 加载迷宫配置
        self._load_maze(maze_id)

        # 当前状态
        self.state = None
        self.steps = 0
        self.max_steps = maze_size * maze_size * 2
        self.previous_distance = None

    def _load_maze(self, maze_id):
        """加载迷宫地图配置"""
        if self.random_obstacles:
            self._generate_random_maze()
            return

        config_path = Path(__file__).parent / "maze_maps.json"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            if str(maze_id) in configs:
                cfg = configs[str(maze_id)]
                self.maze = np.array(cfg["maze"])
                self.maze_size = self.maze.shape[0]  # 更新maze_size为实际大小
                self.start = tuple(cfg["start"])
                self.goal = tuple(cfg["goal"])
                return

        # 默认迷宫: 0=通路, 1=墙壁
        self.maze = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0]
        ])
        self.maze_size = self.maze.shape[0]  # 更新maze_size为实际大小
        self.start = (0, 0)  # 起点：左上角
        self.goal = (4, 4)   # 终点：右下角

    def _generate_random_maze(self):
        """随机生成障碍物分布的迷宫"""
        self.start = (0, 0)
        self.goal = (self.maze_size - 1, self.maze_size - 1)

        while True:
            # 生成随机障碍物
            self.maze = np.zeros((self.maze_size, self.maze_size), dtype=int)
            num_obstacles = int(self.maze_size ** 2 * self.obstacle_ratio)

            # 随机放置障碍物
            positions = []
            for i in range(self.maze_size):
                for j in range(self.maze_size):
                    if (i, j) != self.start and (i, j) != self.goal:
                        positions.append((i, j))

            np.random.shuffle(positions)
            for pos in positions[:num_obstacles]:
                self.maze[pos] = 1

            # 检查是否存在从起点到终点的路径
            if self._path_exists():
                break

    def _path_exists(self):
        """使用BFS检查是否存在从起点到终点的路径"""
        from collections import deque
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)

        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
                return True

            for dx, dy in self.ACTIONS.values():
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.maze_size and 0 <= ny < self.maze_size and
                    (nx, ny) not in visited and self.maze[nx, ny] == 0):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    def _get_observation(self):
        """获取增强的观察状态（归一化坐标 + 8方向障碍物检测）"""
        normalized_pos = [
            self.state[0] / (self.maze_size - 1),
            self.state[1] / (self.maze_size - 1)
        ]

        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        obstacles = []
        for dx, dy in directions:
            nx, ny = self.state[0] + dx, self.state[1] + dy
            if (nx < 0 or nx >= self.maze_size or
                ny < 0 or ny >= self.maze_size or
                self.maze[nx, ny] == 1):
                obstacles.append(1.0)
            else:
                obstacles.append(0.0)

        return normalized_pos + obstacles

    def reset(self, seed=None):
        """
        重置环境

        Args:
            seed: 随机种子（Gymnasium 兼容）

        Returns:
            state: 初始状态（归一化坐标 + 8方向障碍物检测）
        """
        if seed is not None:
            np.random.seed(seed)

        if self.random_obstacles:
            self._generate_random_maze()

        self.state = list(self.start)
        self.steps = 0
        self.previous_distance = abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])
        return self._get_observation()

    def step(self, action):
        """
        执行一步动作

        Args:
            action: 动作编号 (0-3)

        Returns:
            next_state: 下一状态（归一化坐标 + 8方向障碍物检测）
            reward: 奖励值
            done: 是否结束
        """
        self.steps += 1

        dx, dy = self.ACTIONS[action]
        new_x = self.state[0] + dx
        new_y = self.state[1] + dy

        if not (0 <= new_x < self.maze_size and 0 <= new_y < self.maze_size):
            reward = -5
            done = self.steps >= self.max_steps
            return self._get_observation(), reward, done

        if self.maze[new_x, new_y] == 1:
            reward = -5
            done = self.steps >= self.max_steps
            return self._get_observation(), reward, done

        self.state = [new_x, new_y]
        current_distance = abs(self.state[0] - self.goal[0]) + abs(self.state[1] - self.goal[1])

        if tuple(self.state) == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1
            if current_distance < self.previous_distance:
                reward += 0.1
            elif current_distance > self.previous_distance:
                reward -= 0.1
            done = self.steps >= self.max_steps

        self.previous_distance = current_distance
        return self._get_observation(), reward, done

    def render(self, mode='text'):
        """
        渲染当前迷宫状态

        Args:
            mode: 渲染模式 ('text' 或 'array')
        """
        if mode == 'array':
            return self._get_render_array()

        # 文本模式
        symbols = {
            'wall': '█',
            'path': '·',
            'agent': 'A',
            'goal': 'G',
            'start': 'S'
        }

        print("\n" + "=" * (self.maze_size * 2 + 1))
        for i in range(self.maze_size):
            row = ""
            for j in range(self.maze_size):
                if [i, j] == self.state:
                    row += symbols['agent'] + " "
                elif (i, j) == self.goal:
                    row += symbols['goal'] + " "
                elif (i, j) == self.start and self.state != list(self.start):
                    row += symbols['start'] + " "
                elif self.maze[i, j] == 1:
                    row += symbols['wall'] + " "
                else:
                    row += symbols['path'] + " "
            print(row)
        print("=" * (self.maze_size * 2 + 1))
        print(f"位置: {self.state}, 步数: {self.steps}")

    def _get_render_array(self):
        """返回用于可视化的数组"""
        render = self.maze.copy().astype(float)
        # 标记特殊位置
        render[self.start] = 0.3  # 起点
        render[self.goal] = 0.7   # 终点
        if self.state:
            render[self.state[0], self.state[1]] = 0.5  # 智能体
        return render

    def get_valid_actions(self):
        """获取当前状态下的有效动作列表"""
        valid = []
        for action, (dx, dy) in self.ACTIONS.items():
            new_x = self.state[0] + dx
            new_y = self.state[1] + dy
            if (0 <= new_x < self.maze_size and
                0 <= new_y < self.maze_size and
                self.maze[new_x, new_y] == 0):
                valid.append(action)
        return valid


if __name__ == "__main__":
    # 测试代码
    print("=== 测试固定迷宫 ===")
    env = MazeEnv()
    state = env.reset()
    print("初始状态:", state)
    env.render()

    print("\n=== 测试随机迷宫 ===")
    env_random = MazeEnv(random_obstacles=True, obstacle_ratio=0.2)
    state = env_random.reset()
    print("随机迷宫:")
    env_random.render()

    # 随机走几步
    import random
    for _ in range(5):
        action = random.randint(0, 3)
        next_state, reward, done = env.step(action)
        print(f"\n动作: {env.ACTION_NAMES[action]}, 奖励: {reward}, 完成: {done}")
        env.render()
        if done:
            break
