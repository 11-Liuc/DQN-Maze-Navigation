# 基于 DQN 的迷宫寻路智能体

本项目是机器学习课程设计的实现，使用深度强化学习算法（DQN）解决离散空间的迷宫寻路问题。

## 项目进度

- [x] 项目初始化与分工
- [x] 自定义迷宫环境开发（兼容 Gymnasium 接口）
- [x] DQN 算法实现（含经验回放、目标网络）
- [x] 模型训练与调参
- [x] 可视化与分析工具
- [ ] 实验报告与 PPT 完成

## 环境配置

### 1. 创建 Conda 虚拟环境

```bash
# 创建环境
conda create -n dqn-maze python=3.10 -y

# 激活环境
conda activate dqn-maze
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; import matplotlib; import pandas; print('环境配置成功!')"
```

## 快速开始

### 训练模型

```bash
# 基础训练（10x10迷宫，1000回合）
python main.py --mode train

# 使用随机障碍物训练（提升泛化能力）
python main.py --mode train --random_obstacles

# 5x5迷宫训练
python main.py --mode train --maze_size 5 --episodes 500

# 自定义训练参数
python main.py --mode train --maze_size 10 --random_obstacles --episodes 1000 --lr 0.001

# 使用固定迷宫配置（0-3）
python main.py --mode train --maze_id 1 --maze_size 5
```

### 测试模型

```bash
# 使用训练好的模型测试
python main.py --mode test --model outputs/best_model.pth

# 测试泛化能力（在新的随机地图上）
python main.py --mode test --maze_size 10 --random_obstacles --model outputs/best_model.pth --test_episodes 20

# 运行更多测试回合
python main.py --mode test --model outputs/best_model.pth --test_episodes 50
```

### 可视化

```bash
# 显示迷宫结构
python main.py --mode visualize

# 可视化训练好的智能体轨迹
python main.py --mode visualize --model outputs/best_model.pth

# 动画展示
python main.py --mode visualize --model outputs/best_model.pth --animate

# 保存为 GIF
python main.py --mode visualize --model outputs/best_model.pth --animate --save_gif outputs/demo.gif
```

### 训练曲线分析

```bash
# 绘制训练奖励曲线（需先训练）
python main.py --mode plot

# 指定日志文件
python main.py --mode plot --csv training_logs.csv
```

### 超参数调优

```bash
# 运行超参数对比实验（耗时较长）
python main.py --mode tuning
```

## 目录结构

```
DQN-Maze-Navigation/
├── main.py                 # 主入口
├── requirements.txt        # 依赖包
├── README.md              # 项目说明
├── Contribution.md        # 小组分工表
├── docs/                  # 实验报告与汇报材料
│   └── 题目5-基于DQN的迷宫寻路智能体.docx
├── src/                   # 核心源代码
│   ├── maze_env.py        # 迷宫环境（兼容 Gymnasium）
│   ├── maze_maps.json     # 迷宫地图配置
│   ├── dqn_model.py       # DQN 神经网络
│   ├── dqn_agent.py       # DQN 智能体
│   ├── memory.py          # 经验回放缓冲区
│   ├── train_manager.py   # 训练管理器
│   ├── log_analysis.py    # 日志分析与曲线绘制
│   ├── visualizer.py      # 可视化模块
│   └── hyperparameter_tuning.py  # 超参数调优
└── outputs/               # 训练输出（自动生成）
    ├── best_model.pth     # 最佳模型
    ├── final_model.pth    # 最终模型
    ├── training_logs.csv  # 训练日志
    └── training_curves.png # 训练曲线图
```

## 技术实现

### 环境设计

- **迷宫规模**: 支持动态大小（默认 10x10，可配置 5x5 或其他）
- **状态空间**: 归一化坐标 [x/(size-1), y/(size-1)]，范围 [0, 1]，2 维
- **动作空间**: 上(0)、下(1)、左(2)、右(3)，共 4 个动作
- **奖励函数**:
  - 到达终点: +10
  - 碰撞墙壁/边界: -5
  - 每步惩罚: -0.1
- **随机障碍物**: 支持随机生成障碍物（约 20%），提升泛化能力

### DQN 网络结构

```
输入层 (2) → 全连接层 (128, ReLU) → 全连接层 (128, ReLU) → 输出层 (4)
```

### 核心特性

- **经验回放**: 容量 10000，批量大小 64
- **目标网络**: 每 100 步更新一次
- **ε-贪婪策略**: 初始 ε=1.0，按回合衰减，最终 ε=0.01
- **归一化状态表示**: 使用 [0, 1] 范围的相对坐标，提升跨地图泛化能力
- **随机地图训练**: 每回合生成新的障碍物布局，避免记忆固定路径

### 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 迷宫大小 | 10x10 | 可配置为 5x5 或其他 |
| 训练回合数 | 1000 | 10x10 迷宫推荐值 |
| 学习率 (lr) | 0.001 | Adam 优化器学习率 |
| 折扣因子 (γ) | 0.99 | 未来奖励折扣 |
| ε 衰减率 | 0.995 | 每回合 ε = ε × 0.995 |
| 经验池容量 | 10000 | 存储历史经验 |
| 批量大小 | 64 | 每次学习采样数 |
| 目标网络更新 | 100 步 | 同步 target network |

## 预期成果

- [x] 自定义迷宫环境的 Python 代码（含状态定义、奖励函数）
- [x] DQN 模型的 PyTorch 实现（含经验回放、目标网络更新）
- [x] 训练过程的奖励曲线（每 100 步的平均奖励）
- [x] 智能体移动轨迹图
- [ ] 超参数调优报告（说明 γ、学习率、ε 衰减策略对收敛速度的影响）

## 常见问题

### 中文显示乱码

程序已自动配置中文字体支持：
- macOS: Arial Unicode MS / PingFang SC
- Windows: SimHei / Microsoft YaHei
- Linux: WenQuanYi Micro Hei

### 训练效果不好

1. 增加训练回合数: `--episodes 1000`
2. 调整 ε 衰减率: `--epsilon_decay 0.99`
3. 运行超参数调优: `--mode tuning`

---

*详细分工请查看 [Contribution.md](./Contribution.md)*