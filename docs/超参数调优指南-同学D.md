# 超参数调优操作指南（同学D）

本文档指导如何进行 DQN 超参数调优实验，并撰写调优报告。

## 一、你的任务清单

根据 Contribution.md，你需要完成：

1. **执行超参数调优**：对比学习率、γ、ε衰减策略的影响
2. **记录训练数据**：保存每组实验的 CSV 日志
3. **绘制奖励曲线**：每 100 步的平均奖励曲线
4. **撰写调优报告**：分析不同超参数对收敛速度的影响

## 二、环境准备

```bash
# 激活环境
conda activate dqn-maze

# 确认在项目根目录
cd DQN-Maze-Navigation
```

## 三、超参数调优实验

### 方法一：使用自动调优脚本（推荐）

运行以下命令，自动执行所有对比实验：

```bash
python main.py --mode tuning
```

这会自动运行 3 组实验，每组对比 3 种配置，共 9 次训练。

**注意**: 可以指定迷宫大小和是否使用随机障碍物：
```bash
# 在10x10随机地图上调优
python main.py --mode tuning --maze_size 10 --random_obstacles
```

**输出文件**（保存在 `outputs/` 目录）：
- `tuning_learning_rate.png` - 学习率对比图
- `tuning_gamma.png` - 折扣因子对比图
- `tuning_epsilon_decay.png` - ε衰减策略对比图
- `hyperparameter_report.md` - 自动生成的报告框架

### 方法二：手动逐个实验（更灵活）

如果需要更细致的控制，可以手动运行每组实验：

#### 实验1：学习率对比

```bash
# lr = 0.0001（较小）
python main.py --mode train --lr 0.0001 --episodes 1000
mv outputs/training_logs.csv outputs/logs_lr_0.0001.csv

# lr = 0.001（默认）
python main.py --mode train --lr 0.001 --episodes 1000
mv outputs/training_logs.csv outputs/logs_lr_0.001.csv

# lr = 0.01（较大）
python main.py --mode train --lr 0.01 --episodes 1000
mv outputs/training_logs.csv outputs/logs_lr_0.01.csv
```

#### 实验2：折扣因子 γ 对比

```bash
# gamma = 0.9
python main.py --mode train --gamma 0.9 --episodes 1000
mv outputs/training_logs.csv outputs/logs_gamma_0.9.csv

# gamma = 0.95
python main.py --mode train --gamma 0.95 --episodes 1000
mv outputs/training_logs.csv outputs/logs_gamma_0.95.csv

# gamma = 0.99（默认）
python main.py --mode train --gamma 0.99 --episodes 1000
mv outputs/training_logs.csv outputs/logs_gamma_0.99.csv
```

#### 实验3：ε衰减策略对比

```bash
# 快速衰减 decay = 0.95
python main.py --mode train --epsilon_decay 0.95 --episodes 1000
mv outputs/training_logs.csv outputs/logs_decay_0.95.csv

# 默认衰减 decay = 0.995
python main.py --mode train --epsilon_decay 0.995 --episodes 1000
mv outputs/training_logs.csv outputs/logs_decay_0.995.csv

# 慢速衰减 decay = 0.999
python main.py --mode train --epsilon_decay 0.999 --episodes 1000
mv outputs/training_logs.csv outputs/logs_decay_0.999.csv
```

## 四、绘制训练曲线

对每个保存的 CSV 文件绘制曲线：

```bash
python main.py --mode plot --csv outputs/logs_lr_0.001.csv
```

## 五、需要保存的数据

请确保保存以下文件用于报告：

| 文件 | 说明 |
|------|------|
| `outputs/logs_lr_*.csv` | 学习率实验的训练日志 |
| `outputs/logs_gamma_*.csv` | γ 实验的训练日志 |
| `outputs/logs_decay_*.csv` | ε衰减实验的训练日志 |
| `outputs/tuning_*.png` | 对比曲线图 |
| 截图 | 终端输出的关键数据 |

## 六、报告撰写要点

### 6.1 报告结构建议

```
4. 实验与结果分析
   4.1 实验设置
       - 实验环境（硬件、软件版本）
       - 基准超参数配置
   4.2 学习率对比实验
       - 实验配置表
       - 奖励曲线对比图
       - 结果分析
   4.3 折扣因子 γ 对比实验
       - 实验配置表
       - 奖励曲线对比图
       - 结果分析
   4.4 ε衰减策略对比实验
       - 实验配置表
       - 奖励曲线对比图
       - 结果分析
   4.5 最优超参数组合
```

### 6.2 需要记录的指标

从 CSV 文件中提取以下指标：

| 指标 | 说明 | 如何获取 |
|------|------|----------|
| 收敛回合 | 首次达到稳定高奖励的回合 | 观察曲线拐点 |
| 最终平均奖励 | 最后 100 回合的平均奖励 | CSV 最后 100 行求平均 |
| 最佳奖励 | 训练过程中的最高奖励 | CSV 中 total_reward 最大值 |
| 成功率 | 最后 100 回合到达终点的比例 | 奖励 > 0 的比例 |

### 6.3 分析要点

**学习率分析**：
- 学习率过大：训练不稳定，奖励波动大
- 学习率过小：收敛速度慢
- 适中学习率：稳定且快速收敛

**折扣因子 γ 分析**：
- γ 接近 1：更重视长期奖励，适合需要规划的任务
- γ 较小：更重视即时奖励，可能导致短视行为

**ε衰减策略分析**：
- 快速衰减：早期探索不足，可能陷入局部最优
- 慢速衰减：探索充分但收敛较慢
- 适中衰减：平衡探索与利用

## 七、常见问题

### Q1: 训练时间太长怎么办？
可以减少训练回合数或使用更小的迷宫：
```bash
# 减少回合数
python main.py --mode train --episodes 500

# 使用5x5迷宫（训练更快）
python main.py --mode train --maze_size 5 --episodes 500
```

### Q2: 如何对比多个 CSV 文件的曲线？
可以用 Python 脚本读取多个 CSV 并绘制在同一张图上，参考 `src/hyperparameter_tuning.py` 中的 `plot_comparison` 函数。

### Q3: 实验结果不稳定怎么办？
DQN 训练有随机性，建议每组配置运行 2-3 次取平均。

### Q4: 随机障碍物训练有什么好处？
使用 `--random_obstacles` 可以让智能体学习通用策略而非记忆固定路径，提升泛化能力。

