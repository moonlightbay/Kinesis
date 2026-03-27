# Kinesis Simplified: Musculoskeletal Imitation Learning Framework

这个仓库已经被收敛为一条单一主线:

- 只保留肌骨模型的动作模仿学习
- 只保留单专家策略 `PolicyLattice`
- 只保留单任务智能体 `AgentIM`
- 只保留训练、评估和单条动作播放所需的配置与脚本

仓库中已经移除的内容:

- `ball-kick`
- `directional`
- `target-reach`
- `t2m`
- Mixture of Experts (MoE)
- negative mining / 多专家递进训练

详细架构说明见 [docs/Kinesis项目深度学习指南.md](docs/Kinesis项目深度学习指南.md)。

## 1. 保留下来的训练链路

当前主线如下:

```text
src/run.py
  -> AgentIM
  -> AgentHumanoid
  -> PPO
  -> PolicyLattice + Value
  -> MyoLegsIm
  -> KinesisCore
  -> MuJoCo musculoskeletal model
```

也就是说，当前仓库只做一件事:

> 用 PPO 训练一个单专家策略，让肌骨模型去模仿 KIT/AMASS 动作。

## 2. 安装

Linux / CUDA:

```bash
conda create -n kinesis python=3.8
conda activate kinesis
pip install -r requirements.txt
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

MacOS:

```bash
conda create -n kinesis python=3.8
conda activate kinesis
pip install -r macos_requirements.txt
conda install -c conda-forge lxml
```

## 3. 数据准备

### 3.1 SMPL

1. 从 https://smpl.is.tue.mpg.de/ 下载 `SMPL_NEUTRAL.pkl`
2. 放到 `data/smpl/SMPL_NEUTRAL.pkl`

### 3.2 KIT / AMASS 动作集转换

将 AMASS 中 KIT 数据集转换为本项目使用的 `joblib` 动作字典:

```bash
python src/utils/convert_kit.py --path <path_to_amass_kit>
```

转换结果:

- `data/kit_train_motion_dict.pkl`
- `data/kit_test_motion_dict.pkl`

### 3.3 初始姿态缓存

为了让每段动作能稳定地从参考动作附近启动，需要先生成初始姿态缓存。

示例:

```bash
python src/utils/initial_pose.py --config-name config_legs.yaml
python src/utils/initial_pose.py --config-name config_legs.yaml run=eval_run_legs
```

如果你使用其他模型:

```bash
python src/utils/initial_pose.py --config-name config_legs_abs.yaml
python src/utils/initial_pose.py --config-name config_legs_back.yaml
```

输出位置:

- `data/initial_pose/<model>/initial_pose_train.pkl`
- `data/initial_pose/<model>/initial_pose_test.pkl`

### 3.4 资源与预训练模型

下载资源:

```bash
pip install huggingface_hub
python src/utils/download_assets.py --branch kinesis-2.0
```

下载预训练模型:

```bash
python src/utils/download_models.py
```

## 4. 训练

推荐直接使用简化后的训练脚本:

```bash
bash scripts/train-imitation.sh --model legs --exp_name my_run --num_threads 8
```

也可以直接调用入口:

```bash
python src/run.py --config-name config_legs.yaml exp_name=my_run run.num_threads=8
```

可选模型:

- `legs`
- `legs_abs`
- `legs_back`

## 5. 评估

批量评估整套 train/test 动作:

```bash
bash scripts/kit-locomotion.sh --model legs --dataset test --exp_name my_run
```

这个脚本会走 `eval_policy()`，输出:

- success rate
- MPJPE
- frame coverage

## 6. Play

播放单条动作并可视化:

```bash
bash scripts/play-imitation.sh --model legs --dataset test --motion_id 0 --exp_name my_run
```

常用参数:

- `--dataset train|test`
- `--motion_id <index>`
- `--headless False|True`

## 7. 剩余目录

```text
cfg/
  config_legs*.yaml         # 三种肌骨模型入口
  env/env_im.yaml           # imitation 环境超参数
  learning/im_mlp.yaml      # PPO + PolicyLattice 超参数
  run/train_run_*.yaml      # 训练运行参数
  run/eval_run_*.yaml       # 批量评估参数
  run/play_run_*.yaml       # 单条播放参数

src/
  run.py
  agents/
  env/
  learning/
  KinesisCore/
  utils/convert_kit.py
  utils/initial_pose.py
```

## 8. 默认策略与算法

当前固定为:

- 策略: `PolicyLattice`
- 值函数: `Value(MLP)`
- 强化学习算法: `PPO`
- 任务: `MyoLegsIm`

默认超参数可在以下位置查看:

- `cfg/learning/im_mlp.yaml`
- `cfg/env/env_im.yaml`

## 9. Checkpoint

模型默认保存在:

```text
data/trained_models/<model>/<exp_name>/
```

其中:

- `model.pth` 是最新 checkpoint
- `model_00001000.pth` 这类文件是定期保存的历史 checkpoint

## 10. 推荐阅读顺序

1. `src/run.py`
2. `src/agents/agent_humanoid.py`
3. `src/agents/agent_im.py`
4. `src/env/myolegs_env.py`
5. `src/env/myolegs_im.py`
6. `src/KinesisCore/kinesis_core.py`
7. `src/learning/policy_lattice.py`
8. `src/agents/agent_ppo.py`

如果你们后续要继续在这个框架上扩展新的 imitation 变体，建议先从 `docs/Kinesis项目深度学习指南.md` 开始。
