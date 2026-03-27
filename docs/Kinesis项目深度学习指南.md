# Kinesis 项目深度学习指南

这份文档对应的是当前已经简化后的仓库版本:

- 只保留肌骨模型模仿学习
- 只保留单专家策略
- 只保留 `AgentIM + MyoLegsIm + PPO + PolicyLattice`

如果你在代码里看到和本文不一致的旧分支，请以当前仓库文件为准。旧的下游任务和 MoE 架构已经移除。

## 1. 当前架构总览

当前训练链路可以概括成一句话:

> 用离线动作数据构造参考轨迹，在 MuJoCo 肌骨环境里做 imitation reward，再用 PPO 训练一个单专家策略去跟踪动作。

执行链路:

```text
Hydra config
  -> src/run.py
  -> AgentIM
  -> AgentHumanoid
  -> MyoLegsIm
  -> KinesisCore
  -> PolicyLattice + Value
  -> PPO
```

## 2. 现在保留下来的核心文件

### 入口

- `src/run.py`

### Agent

- `src/agents/agent.py`
- `src/agents/agent_pg.py`
- `src/agents/agent_ppo.py`
- `src/agents/agent_humanoid.py`
- `src/agents/agent_im.py`

### Environment

- `src/env/myolegs_base_env.py`
- `src/env/myolegs_env.py`
- `src/env/myolegs_task.py`
- `src/env/myolegs_im.py`

### Learning

- `src/learning/policy.py`
- `src/learning/policy_lattice.py`
- `src/learning/critic.py`
- `src/learning/memory.py`
- `src/learning/trajbatch.py`
- `src/learning/learning_utils.py`

### Motion Core

- `src/KinesisCore/kinesis_core.py`
- `src/KinesisCore/forward_kinematics.py`

### Data Preparation

- `src/utils/convert_kit.py`
- `src/utils/initial_pose.py`

## 3. 从动捕数据到参考轨迹

### 3.1 原始来源

当前项目使用的是 AMASS 中的 KIT locomotion 数据。

### 3.2 转换入口

使用:

```bash
python src/utils/convert_kit.py --path <path_to_amass_kit>
```

这个脚本会做以下事情:

1. 遍历 KIT `.npz`
2. 读取 `trans`、`poses`、`mocap_framerate`
3. 将动作统一下采样到 30 FPS
4. 整理成 24 关节的 SMPL 轴角表示 `pose_aa`
5. 生成训练集和测试集两个 `joblib` 字典

输出文件:

- `data/kit_train_motion_dict.pkl`
- `data/kit_test_motion_dict.pkl`

每条 motion 里最关键的字段有:

- `pose_aa`
- `trans_orig`
- `fps`

## 4. 初始姿态缓存

模仿学习不是直接把身体扔到默认站姿开始跑，而是尽量从参考动作附近启动。

这部分由:

```text
src/utils/initial_pose.py
```

负责生成。

### 4.1 生成逻辑

脚本会逐条 motion 遍历不同起始时间点，然后调用环境 reset，记录每个起始时刻对应的 `initial_pose`。

输出文件:

- `data/initial_pose/<model>/initial_pose_train.pkl`
- `data/initial_pose/<model>/initial_pose_test.pkl`

### 4.2 环境里怎么使用

在 `MyoLegsIm.initialize_motion_state()` 中:

1. 根据当前 motion id 和 start time 查初始姿态缓存
2. 如果找到了，就直接把缓存姿态写入 `mj_data.qpos`
3. 如果找不到，就调用 `compute_initial_pose()` 在线优化一个姿态

## 5. KinesisCore: 参考轨迹生成器

`src/KinesisCore/kinesis_core.py` 是动作参考系统的核心。

### 5.1 它做什么

1. 从 `joblib` motion dict 读数据
2. 调用 `ForwardKinematics` 把 SMPL 动作转成可查询的刚体轨迹
3. 缓存下面这些量:
   - `xpos`
   - `xquat`
   - `body_vel`
   - `body_ang_vel`
   - `qpos`
   - `qvel`
4. 在环境每个 step 按当前时间戳返回参考状态

### 5.2 为什么它重要

强化学习环境本身只知道 MuJoCo 当前状态，不知道“应该模仿成什么样”。

KinesisCore 负责把离线动作数据变成在线可查询的参考目标。

## 6. 训练入口如何启动

入口文件是:

```text
src/run.py
```

它负责:

1. 读取 Hydra 配置
2. 初始化 `wandb`
3. 设定随机种子和 deterministic 模式
4. 构建 `AgentIM`
5. 根据 `cfg.run.test` 选择训练、评估还是播放

### 6.1 三种模式

#### 训练

当 `cfg.run.test=False`:

```text
agent.optimize_policy()
```

#### 批量评估

当 `cfg.run.test=True` 且 `cfg.run.im_eval=True`:

```text
agent.eval_policy()
```

#### Play

当 `cfg.run.test=True` 且 `cfg.run.im_eval=False`:

```text
agent.run_policy()
```

## 7. Agent 链路

继承关系:

```text
Agent
  -> AgentPG
  -> AgentPPO
  -> AgentHumanoid
  -> AgentIM
```

### 7.1 `Agent`

负责通用 RL rollout:

- 多进程采样
- 经验存储 `Memory`
- 打包 `TrajBatch`
- observation/action 预处理

### 7.2 `AgentPG`

负责:

- 计算 value
- 估计 advantage / returns
- 调用策略更新

### 7.3 `AgentPPO`

负责:

- 计算旧策略 log-prob
- PPO clipped surrogate loss
- value loss
- 梯度裁剪

### 7.4 `AgentHumanoid`

负责:

- 创建环境
- 创建策略和值函数
- 创建优化器
- checkpoint 读写
- 训练主循环

### 7.5 `AgentIM`

负责 imitation-specific 行为:

- 创建 `MyoLegsIm`
- 训练时周期性重采样 motions
- 批量评估 motion imitation 成功率、MPJPE、frame coverage

## 8. Policy / Value / PPO

### 8.1 策略

当前只保留:

```text
src/learning/policy_lattice.py
```

它是一个单专家连续动作策略:

1. `RunningNorm` 对 observation 做归一化
2. `MLP` 提取特征
3. 输出动作均值 `action_mean`
4. 构造带相关性的高斯分布

### 8.2 值函数

值函数在 `AgentHumanoid.setup_value()` 中构造:

- 主干是 `MLP`
- 外面包一层 `Value`

### 8.3 PPO

PPO 关键参数在 `cfg/learning/im_mlp.yaml`:

- `opt_num_epochs: 10`
- `min_batch_size: 51200`
- `policy_grad_clip: 25`
- `gamma: 0.99`
- `tau: 0.95`
- `clip_epsilon: 0.2`
- `policy_lr: 5e-5`
- `value_lr: 3e-4`

网络规模:

```yaml
mlp:
  units: [2048, 1536, 1024, 1024, 512, 512]
  activation: silu
```

## 9. 环境链路

继承关系:

```text
BaseEnv
  -> MyoLegsEnv
  -> MyoLegsTask
  -> MyoLegsIm
```

### 9.1 `BaseEnv`

提供 Gym 风格接口:

- `reset`
- `step`
- `render`
- `create_sim`

### 9.2 `MyoLegsEnv`

负责 MuJoCo 肌骨环境底层逻辑:

- 加载 XML
- 定义 observation/action space
- 计算 proprioception
- 把策略输出变成肌肉控制
- 推进仿真

### 9.3 `MyoLegsTask`

把环境拆成:

- 本体观测
- 任务观测

最终 observation:

```text
obs = proprioception + task_obs
```

### 9.4 `MyoLegsIm`

这是 imitation learning 的任务层，负责:

- 读取参考动作
- reset 到合适的动作起点
- 计算 imitation observation
- 计算 imitation reward
- 决定 terminate / truncate

## 10. Obs: 当前保留的观测结构

### 10.1 Proprioception

由 `cfg/run/*.yaml` 中的 `proprioceptive_inputs` 控制，默认包括:

- `root_height`
- `root_tilt`
- `local_body_pos`
- `local_body_rot`
- `local_body_vel`
- `local_body_ang_vel`
- `feet_contacts`

这些在 `MyoLegsEnv.compute_proprioception()` 里构造。

### 10.2 Task Observation

由 `MyoLegsIm.compute_task_obs()` 构造，默认包括:

- `diff_local_body_pos`
- `diff_local_vel`
- `local_ref_body_pos`

它们分别表示:

- 当前身体相对参考动作的局部位置误差
- 当前身体相对参考动作的局部速度误差
- 参考身体在当前 root 局部坐标系下的位置

## 11. Step 流程

每一步的主要执行顺序如下:

```text
Agent 选动作
  -> env.step(action)
  -> physics_step(action)
  -> MuJoCo rollout
  -> compute_observations()
  -> compute_reward()
  -> compute_reset()
  -> 返回 obs, reward, terminated, truncated, info
```

其中 `physics_step()` 在 `MyoLegsEnv` 中完成。

## 12. Action 与控制模式

当前保留两种控制模式:

- `PD`
- `direct`

### 12.1 PD

流程:

```text
action -> target muscle length -> force -> activation
```

相关函数:

- `action_to_target_length`
- `target_length_to_force`
- `target_length_to_activation`

### 12.2 direct

流程:

```text
action in [-1, 1] -> activation in [0, 1]
```

## 13. Reward

环境奖励由 `MyoLegsIm.compute_reward()` 计算。

### 13.1 imitation reward

来自 `compute_imitation_reward()`，包含两项:

- `r_body_pos`
- `r_vel`

默认参数在 `cfg/env/env_im.yaml`:

```yaml
k_pos: 200
k_vel: 5
w_pos: 0.6
w_vel: 0.2
```

### 13.2 upright reward

鼓励 root 保持直立:

```yaml
w_upright: 0.1
```

### 13.3 energy reward

对能耗进行正则:

```yaml
k_energy: 0.05
w_energy: 0.1
```

### 13.4 总奖励

可以理解成:

```text
reward = imitation + upright + energy
```

## 14. Reset / Termination / Truncation

### 14.1 Termination

如果当前身体和参考动作偏差过大，就提前失败。

对应参数:

```yaml
termination_distance: 0.15
```

### 14.2 Truncation

如果动作完整播放结束，则本回合正常结束。

### 14.3 训练中的意义

这意味着当前策略学的不是“站住不倒”，而是:

> 在肌骨动力学约束下，尽可能长时间地跟住整段参考动作。

## 15. Checkpoint

checkpoint 在 `AgentHumanoid` 中统一管理。

保存规则:

- `model.pth`: 最新模型
- `model_XXXXXXXX.pth`: 周期保存的历史模型

相关超参数:

- `save_curr_frequency: 50`
- `save_frequency: 1000`

输出目录:

```text
data/trained_models/<model>/<exp_name>/
```

## 16. 训练、评估、播放命令

### 16.1 训练

```bash
bash scripts/train-imitation.sh --model legs --exp_name my_run --num_threads 8
```

### 16.2 批量评估

```bash
bash scripts/kit-locomotion.sh --model legs --dataset test --exp_name my_run
```

### 16.3 单条动作播放

```bash
bash scripts/play-imitation.sh --model legs --dataset test --motion_id 0 --exp_name my_run
```

### 16.4 直接用 Python 入口

训练:

```bash
python src/run.py --config-name config_legs.yaml exp_name=my_run run.num_threads=8
```

评估:

```bash
python src/run.py --config-name config_legs.yaml exp_name=my_run epoch=-1 run=eval_run_legs
```

播放:

```bash
python src/run.py --config-name config_legs.yaml exp_name=my_run epoch=-1 run=play_run_legs
```

## 17. 剩余配置文件说明

### 17.1 顶层配置

- `cfg/config_legs.yaml`
- `cfg/config_legs_abs.yaml`
- `cfg/config_legs_back.yaml`

职责:

- 选择模型
- 绑定 env / learning / run 三类子配置
- 定义输出目录

### 17.2 环境超参数

- `cfg/env/env_im.yaml`

职责:

- 仿真步长
- 控制频率
- reward 权重
- termination 距离
- motion 重采样周期

### 17.3 学习超参数

- `cfg/learning/im_mlp.yaml`

职责:

- PPO 超参数
- 策略/值函数学习率
- `PolicyLattice` 主干 MLP 规模

### 17.4 运行配置

训练:

- `cfg/run/train_run_legs.yaml`
- `cfg/run/train_run_legs_abs.yaml`
- `cfg/run/train_run_legs_back.yaml`

评估:

- `cfg/run/eval_run_legs.yaml`
- `cfg/run/eval_run_legs_abs.yaml`
- `cfg/run/eval_run_legs_back.yaml`

播放:

- `cfg/run/play_run_legs.yaml`
- `cfg/run/play_run_legs_abs.yaml`
- `cfg/run/play_run_legs_back.yaml`

## 18. 现在最推荐的阅读顺序

如果你们是为了后续二次开发，建议按下面顺序读:

1. `src/run.py`
2. `cfg/config_legs.yaml`
3. `cfg/learning/im_mlp.yaml`
4. `cfg/env/env_im.yaml`
5. `cfg/run/train_run_legs.yaml`
6. `src/agents/agent_humanoid.py`
7. `src/agents/agent_im.py`
8. `src/env/myolegs_env.py`
9. `src/env/myolegs_im.py`
10. `src/KinesisCore/kinesis_core.py`
11. `src/learning/policy_lattice.py`
12. `src/agents/agent_ppo.py`

## 19. 适合继续扩展的方向

当前这个简化版最适合做以下事情:

- 研究 reward 设计对 imitation 的影响
- 替换策略网络结构，但仍保持单专家
- 增加新的 proprioceptive / task observation
- 对不同肌骨模型做横向比较
- 在 imitation 主线稳定后，再独立开发新的任务

如果后面你们要继续扩展，我建议始终先保持这条主线稳定:

```text
数据转换 -> 初始姿态 -> imitation env -> PPO -> checkpoint -> eval/play
```

这样每次改动都容易定位问题出在哪一层。
