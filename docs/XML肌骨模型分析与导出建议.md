# Kinesis 项目 XML 肌骨模型分析与导出建议

## 1. 文档目标

这份文档面向两个任务：

1. 梳理项目中 `data/xml` 下每个 `xml` 文件的内容、作用和依赖关系。
2. 给出后续继续研究肌骨模型建模的建议，以及如何把你自己设计的模型 `xml` 导出到其他代码库。

结论先行：

- 这个项目里的 `xml` 不是“一个大文件”，而是明显的分层组合体系：`入口模型` + `资产定义` + `骨架链` + `肌腱` + `肌肉执行器` + `场景`。
- Kinesis 训练时真正直接加载的是少数几个顶层入口文件；其余文件主要作为被 `include` 的组件。
- 项目同时还保留了一条 `SMPL -> MJCF/XML` 的自动生成链路，这对你以后导出自定义模型很有帮助。

## 2. 全局结构总览

### 2.1 目录分层

`data/xml` 里的 51 个 `xml` 基本可以分为 8 组：

| 分组 | 主要用途 |
| --- | --- |
| `humanoid_template_local.xml` / `smpl_humanoid.xml` | 供 SMPL 骨架转 MuJoCo/MJCF 使用 |
| `legs/` | 80 肌肉的基础下肢模型，带刚性 torso/head 外形 |
| `legs_abs/` | 86 肌肉版本，在 `legs` 基础上加入简化腹部肌 |
| `legs_back/head/` | 头部子模型 |
| `legs_back/leg/` | 290 肌肉体系中的下肢子模型入口 |
| `legs_back/torso/` | MyoBack 躯干 / 腰背 / 腹部 / 外骨骼相关模型 |
| `legs_back/scene/` | 通用场景、地面、logo、四边形场景 |
| `legs_back/leg_soccer/` | 足球任务版本：人体模型 + 球场 + 足球 |

### 2.2 文件角色分层

从 MuJoCo/MJCF 角度看，这些文件大致分成 6 种角色：

| 角色 | 典型内容 | 典型文件 |
| --- | --- | --- |
| 顶层入口文件 | `<mujoco>`，定义 `worldbody`、场景、根 body、`include` 其他子文件 | `myolegs.xml`、`myolegs_abdomen.xml`、`myotorso.xml` |
| 资产定义文件 | `<asset>`、`<default>`、`<sensor>`、`<equality>`、材质和 mesh 注册 | `myolegs_assets.xml`、`myotorso_assets.xml` |
| 骨架链文件 | `<body>`、`<joint>`、`<site>`、包裹体 `geom` | `myolegs_chain.xml`、`myotorso_chain.xml` |
| 肌腱文件 | `<tendon><spatial>...`，定义肌腱路径 | `myolegs_tendon.xml` |
| 肌肉执行器文件 | `<actuator><general>...`，定义肌肉力学参数 | `myolegs_muscle.xml` |
| 场景文件 | 地板、光照、相机、球场、足球、logo 等 | `myosuite_scene.xml`、`soccer_field.xml` |

### 2.3 当前代码真正加载哪些 XML

项目环境类最终直接调用 `mujoco.MjModel.from_xml_path(cfg.run.xml_path)`，也就是“只加载顶层入口 XML”，其他组件都是靠 `include` 进来的。

当前配置里最关键的 3 个入口文件是：

- `cfg/run/train_run_legs.yaml` -> `data/xml/legs/myolegs.xml`
- `cfg/run/train_run_legs_abs.yaml` -> `data/xml/legs_abs/myolegs_abdomen.xml`
- `cfg/run/train_run_legs_back.yaml` -> `data/xml/legs_back/leg_soccer/myolegs_soccer.xml`

所以如果你后面要“搭完整模型并修改”，优先应该从这 3 条入口链路开始理解。

### 2.4 三条主组合链路

#### `legs` 80 肌肉链路

```text
data/xml/legs/myolegs.xml
  -> myotorsorigid_assets.xml
  -> myolegs_assets.xml
```

说明：这个版本把腿部的 assets、sensor、equality、tendon、muscle actuator 基本都包在 `myolegs_assets.xml` 里，结构最紧凑。

#### `legs_abs` 86 肌肉链路

```text
data/xml/legs_abs/myolegs_abdomen.xml
  -> myotorso_abdomen_assets.xml
  -> myolegs_assets.xml
  -> myolegs_tendon.xml
  -> myolegs_muscle.xml
  -> myotorso_abdomen_chain.xml
  -> myolegs_chain.xml
```

说明：这个版本把“骨架 / 肌腱 / 肌肉执行器”明确拆开了，更适合继续做二次建模。

#### `legs_back` 290 肌肉链路

```text
data/xml/legs_back/leg_soccer/myolegs_soccer.xml
  -> soccer_assets/myohead_simple_assets.xml
  -> soccer_assets/myolegs_assets.xml
  -> soccer_assets/myolegs_muscle.xml
  -> soccer_assets/myolegs_tendon.xml
  -> soccer_assets/myotorso_assets.xml
  -> soccer_assets/myotorso_chain.xml
  -> soccer_assets/myolegs_chain.xml
  -> (可选) soccer_field.xml / soccer_ball.xml
```

说明：这是 Kinesis 里最复杂、最接近“完整身体控制”的版本，下肢 80 肌肉 + 躯干 210 肌肉，总计 290 肌肉。

## 3. 逐文件分析

下面按目录给出每个 `xml` 的内容和作用。

### 3.1 根目录与自动生成相关文件

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/humanoid_template_local.xml` | 一个最小 MJCF 模板，提供基础 `<default>`、地面、空的 `actuator/contact/sensor` 容器。 | 给 SMPL 骨架生成器当模板底板，后续由代码往里面写 body、joint、motor。 |
| `data/xml/smpl_humanoid.xml` | 已经生成好的刚体 humanoid，包含完整人体 body 树、70 个 joint、69 个 motor。 | 给 `SkeletonTree.from_mjcf(...)` 这类流程使用，是 SMPL 动作和 MuJoCo 骨架之间的桥梁。 |

### 3.2 `data/xml/legs/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs/myolegs.xml` | 顶层 80 肌肉下肢模型入口，包含刚性 torso/head 外形、完整 pelvis-root 结构、自由根关节。 | `legs` 模型的主入口文件。适合先熟悉最基础的肌骨控制结构。 |
| `data/xml/legs/myolegs_assets.xml` | 同时包含 `default`、材质、mesh、接触配对、膝关节等式约束、传感器、80 条肌腱和 80 个肌肉 actuator。 | `legs` 版本的核心资源包，几乎把肌骨系统所有非骨架信息集中在一起。 |
| `data/xml/legs/myotorsorigid_assets.xml` | 只定义刚性 torso/head 外观、材质和少量碰撞属性，不含真实躯干肌肉。 | 给 `legs` 模型提供“外形上的上半身”，但不增加躯干肌肉控制复杂度。 |

### 3.3 `data/xml/legs_abs/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs_abs/myolegs_abdomen.xml` | 86 肌肉版本顶层入口，组合腹部/简化躯干资产、腿部资产、肌腱、肌肉和骨架链。 | `legs_abs` 的主入口文件，也是最适合你继续改模的版本之一。 |
| `data/xml/legs_abs/myolegs_assets.xml` | 腿部公共资产层，包含默认参数、mesh、接触配对、膝关节约束、传感器。 | 给 `legs_abs` 提供腿部的“公共非骨架部分”。 |
| `data/xml/legs_abs/myolegs_chain.xml` | 腿部 body/joint/site/wrap 几何链，描述 pelvis 到双腿的解剖结构和肌肉路径点。 | 腿部几何拓扑的主体文件，改骨段长度、关节位置、包裹体时重点看这里。 |
| `data/xml/legs_abs/myolegs_tendon.xml` | 80 条腿部肌腱路径定义，指定每条肌肉经过哪些 `site` / `geom wrap`。 | 改肌肉走向、包裹路径、springlength 时重点修改。 |
| `data/xml/legs_abs/myolegs_muscle.xml` | 80 个 `<general>` 肌肉执行器，包含 `biasprm/gainprm/lengthrange` 等力学参数。 | 改最大力、最佳长度、动力学响应时重点修改。 |
| `data/xml/legs_abs/myotorso_abdomen_assets.xml` | 简化躯干 + 腹部资产，包含腹部 6 条肌腱和 6 个腹部肌肉 actuator，同时引入头部 simple assets。 | 在腿部之外补上最小必要的腹部驱动，让总肌肉数从 80 变为 86。 |
| `data/xml/legs_abs/myotorso_abdomen_chain.xml` | 简化躯干和头部骨架链，定义 sacrum、lumbar、thoracic、neck/head 等 body/site。 | 把腿部模型扩展成“腿 + 简化躯干 + 头”的完整运动学链。 |
| `data/xml/legs_abs/myohead_simple_assets.xml` | 头部简单资产，定义 mesh/material/default。 | 被 abdomen/torso 版本复用，提供头部外形和基础属性。 |
| `data/xml/legs_abs/myohead_rigid_chain.xml` | 刚性头链，定义颈部与头部的刚性几何结构。 | 在不引入复杂头部自由度的前提下，给 torso 链补上头部。 |

### 3.4 `data/xml/legs_back/head/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs_back/head/myohead_simple.xml` | 头部子模型顶层入口，加载通用场景和 head assets/chain。 | 独立测试头部模型时用。 |
| `data/xml/legs_back/head/assets/myohead_simple_assets.xml` | 头部材质、mesh、默认属性。 | 给 head / torso / soccer 模型提供头部资源。 |
| `data/xml/legs_back/head/assets/myohead_simple_chain.xml` | 带简单关节链的头颈结构。 | 独立头部模型使用。 |
| `data/xml/legs_back/head/assets/myohead_rigid_chain.xml` | 刚性头链。 | 给 torso 和 soccer 躯干链挂接刚性头部。 |

### 3.5 `data/xml/legs_back/leg/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs_back/leg/myolegs.xml` | 290 肌肉体系里的下肢入口，加载无 pedestal 场景、刚性 torso 外形、腿部资产、肌腱、肌肉和骨架链。 | “腿 + 刚性躯干外形”版本主入口。 |
| `data/xml/legs_back/leg/myolegs_abdomen.xml` | 与上面类似，但把刚性躯干换成简化 abdomen 躯干。 | “腿 + 简化腹部”版本主入口。 |
| `data/xml/legs_back/leg/assets/myolegs_assets.xml` | 腿部默认参数、mesh、接触、约束、传感器。 | `legs_back` 中腿部的公共资源层。 |
| `data/xml/legs_back/leg/assets/myolegs_chain.xml` | 下肢骨架链、关节、路径点和 wrap 几何。 | `legs_back` 腿部结构主体。 |
| `data/xml/legs_back/leg/assets/myolegs_tendon.xml` | 80 条腿部肌腱路径。 | 定义腿部肌肉路径。 |
| `data/xml/legs_back/leg/assets/myolegs_muscle.xml` | 80 个腿部肌肉 actuator。 | 定义腿部肌力学参数。 |

### 3.6 `data/xml/legs_back/torso/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs_back/torso/myotorso.xml` | 完整 MyoTorso 顶层入口，组合通用场景、躯干 assets 和 chain。 | 独立躯干模型入口，理论上用于单独测试 210 肌肉背部/腰椎模型。 |
| `data/xml/legs_back/torso/myotorso_abdomen.xml` | 简化 abdomen 版本躯干入口。 | 比完整 MyoBack 更轻量，便于与腿部组合。 |
| `data/xml/legs_back/torso/myotorso_rigid.xml` | 刚性躯干入口，只保留躯干几何，不带真实躯干肌。 | 给腿部模型提供刚性上半身。 |
| `data/xml/legs_back/torso/myotorso_exosuit.xml` | 躯干 + 外骨骼版本，包含外骨骼 mesh、weld 约束和外骨骼 tendon。 | 研究背部辅助装置、外骨骼耦合时使用。 |
| `data/xml/legs_back/torso/assets/myotorso_assets.xml` | 完整 MyoBack 资产层，包含材质、躯干 mesh、虚拟关节映射 `equality`、210 条肌腱和 212 个 actuator 相关定义。 | 210 肌肉躯干系统的资源核心。 |
| `data/xml/legs_back/torso/assets/myotorso_chain.xml` | 躯干 body/joint/site 链，从 sacrum 经 lumbar 到 thoracic，并挂接头链。 | MyoBack 的运动学主链。 |
| `data/xml/legs_back/torso/assets/myotorso_rigid_assets.xml` | 刚性躯干的材质和少量外观属性。 | 不引入 210 肌肉时的轻量替代。 |
| `data/xml/legs_back/torso/assets/myotorso_rigid_chain.xml` | 刚性 torso/head 几何链。 | 给 `myolegs.xml` 这类入口提供刚性躯干外形。 |
| `data/xml/legs_back/torso/assets/myotorso_abdomen_assets.xml` | 简化腹部版资产层，含 6 条腹部肌腱和 6 个腹部 actuator。 | 作为完整 MyoBack 与刚性 torso 之间的中间复杂度方案。 |
| `data/xml/legs_back/torso/assets/myotorso_abdomen_chain.xml` | 简化 abdomen 版骨架链。 | 与腿部拼装成 `legs_back/leg/myolegs_abdomen.xml`。 |

### 3.7 `data/xml/legs_back/scene/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs_back/scene/myosuite_scene.xml` | 标准 MyoSuite 室内场景，包含地面、灯光、相机、logo、pedestal。 | 通用展示/调试场景。 |
| `data/xml/legs_back/scene/myosuite_scene_noPedestal.xml` | 去掉 pedestal 的场景版本，地面更适合 locomotion。 | 腿部 locomotion 模型更常用。 |
| `data/xml/legs_back/scene/myosuite_quad.xml` | 更简化的四边形地面/场景。 | 快速测试或替代场景。 |
| `data/xml/legs_back/scene/myosuite_logo.xml` | 单独的 logo 场景对象。 | 纯展示用途，不属于肌骨模型核心。 |

### 3.8 `data/xml/legs_back/leg_soccer/`

| 文件 | 内容分析 | 作用 |
| --- | --- | --- |
| `data/xml/legs_back/leg_soccer/myolegs_soccer.xml` | 足球任务的主入口，加载头、腿、躯干资产，并在 `root` 下拼接 torso + legs 链。默认不包含球场和足球。 | `legs_back` 当前训练配置默认使用的主模型入口。 |
| `data/xml/legs_back/leg_soccer/myolegs_soccer_goalie.xml` | 在主入口基础上额外引入 `soccer_field.xml` 和 `soccer_ball.xml`。 | 守门 / 真实场景交互测试用。 |
| `data/xml/legs_back/leg_soccer/myolegs_soccer_grass.xml` | 同类足球场景变体。 | 不同球场展示或接触设置时使用。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myohead_simple_assets.xml` | 足球版本头部资产。 | 给 soccer 入口提供头部外形。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myohead_rigid_chain.xml` | 足球版本刚性头链。 | 由 torso chain 挂接。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myolegs_assets.xml` | 足球版本腿部资产层。 | 定义腿部 mesh、传感器、约束等。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myolegs_chain.xml` | 足球版本腿部骨架链。 | 定义腿部结构。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myolegs_tendon.xml` | 80 条腿部肌腱。 | 定义腿部肌腱路径。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myolegs_muscle.xml` | 80 个腿部肌肉 actuator。 | 定义腿部肌力学参数。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myotorso_assets.xml` | 足球版本完整躯干资产层，包含 210 肌肉躯干资源。 | 与腿部组合成 290 肌肉全模型。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/myotorso_chain.xml` | 足球版本躯干链，并在链尾挂接头部。 | 与 `myolegs_chain.xml` 一起组成整个人体主链。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/soccer_scene/soccer_field.xml` | 球场平面、球门 mesh、贴图材质。 | 足球任务场景。 |
| `data/xml/legs_back/leg_soccer/soccer_assets/soccer_scene/soccer_ball.xml` | 足球刚体，包含球体材质、质量、摩擦和自由关节。 | 足球任务交互对象。 |

## 4. 你真正应该优先研究哪些文件

如果你的目标是“搭建完整肌骨模型并继续改”，我建议按下面顺序研究：

### 第一优先级：先摸清入口文件

建议按这个顺序：

1. `data/xml/legs_abs/myolegs_abdomen.xml`
2. `data/xml/legs_back/leg/myolegs_abdomen.xml`
3. `data/xml/legs_back/leg_soccer/myolegs_soccer.xml`

原因：

- 这些文件最能体现“完整模型如何被拼起来”。
- 里面能直接看见最关键的 `include` 组合关系。
- `legs_abs` 比 `legs_back` 简单，非常适合作为修改前的过渡版本。

### 第二优先级：再看 4 类可修改核心文件

如果你后面要真正改模型，最常改的是：

| 目标 | 优先改哪个文件 |
| --- | --- |
| 改骨架拓扑、骨段长度、关节位置、site 路径点 | `*_chain.xml` |
| 改肌腱路径、包裹体、springlength | `*_tendon.xml` |
| 改肌肉力学参数 | `*_muscle.xml` 或 `myotorso_assets.xml` 里的 `<general>` |
| 改 mesh / 材质 / 传感器 / contact / equality | `*_assets.xml` |

### 第三优先级：最后再碰场景和任务文件

这些文件不是建模核心，但影响仿真表现：

- `scene/*.xml`
- `soccer_field.xml`
- `soccer_ball.xml`
- `myotorso_exosuit.xml`

## 5. 进一步研究肌骨模型建模的建议

### 5.1 建模顺序建议

不要一开始就从 290 肌肉版本下手。更稳妥的路线是：

1. 用 `legs_abs` 建立“骨架链 / 肌腱 / 肌肉 actuator / 场景”的整体概念。
2. 在 `legs_abs` 上做一次小改动，例如改 1 条肌肉路径、改 1 个关节范围、改 1 个骨段长度。
3. 确认 MuJoCo 能编译、能站立、关键 site 和 tendon 不穿模。
4. 再把同样的修改迁移到 `legs_back`。

这样能明显降低排错成本。

### 5.2 每次改模只改一层

非常建议你一次只改一类文件：

- 只改 `chain`
- 或只改 `tendon`
- 或只改 `muscle`

不要同一轮同时改骨架、路径和肌肉参数，否则很难判断是哪里出了问题。

### 5.3 改模时优先关注的 6 个检查点

1. `joint range` 是否仍然合理，尤其是髋、膝、踝和脊柱。
2. `site` 是否还落在正确骨段上。
3. `wrap geom` 是否导致 tendon 跳变或穿模。
4. `springlength` 和 `lengthrange` 是否还匹配新的几何结构。
5. `equality` 约束是否仍适用于修改后的关节链，尤其是膝关节和躯干虚拟关节映射。
6. 接触体 `geom` 是否与可视 mesh 对齐，否则很容易出现看起来正常、实际碰撞错误的情况。

### 5.4 最值得做的 4 类实验

1. `几何一致性实验`
   先不训练，只在 MuJoCo 里摆关键姿态，检查 tendon 路径、joint range、接触体和 mesh 是否一致。
2. `肌肉参数敏感性实验`
   单独调整 1 条或 1 组肌肉的 `gainprm/biasprm/lengthrange`，观察控制和力输出变化。
3. `约束正确性实验`
   重点检查膝关节多项式约束和躯干虚拟关节映射在大范围动作下是否稳定。
4. `迁移实验`
   先在 `legs_abs` 改，再迁移到 `legs_back`，验证你设计的结构修改是否具有可复用性。

### 5.5 推荐你的建模工作流

我建议固定成下面的流程：

1. 复制一个当前可用的入口模型目录做实验分支。
2. 先改 `chain.xml`。
3. 再同步检查 `tendon.xml` 的路径点。
4. 最后再改 `muscle.xml` 或 `<general>` 参数。
5. 每一步都重新编译 XML，并做一次短时可视化检查。

## 6. 如何把自己设计的模型 XML 导出到其他代码库

这里其实有两种“导出”路径。

### 6.1 路径 A：你已经有手写/改好的 MJCF XML

这是最常见的情况。做法不是只复制一个入口 `xml`，而是把“入口 + 依赖资源”一起打包。

#### 你至少要导出这些内容

- 顶层入口 `xml`
- 所有被 `include` 的子 `xml`
- 所有 `mesh` 文件（`.stl` / `.obj` / `.msh`）
- 所有贴图文件（`.png` 等）

#### 最稳妥的导出方式

1. 先确定目标入口文件，例如 `myolegs_abdomen.xml`。
2. 递归收集它依赖的所有 `include file=...`。
3. 收集这些文件里 `mesh file=...`、`texture file=...`、`fileup/filedown/...` 引用到的资源。
4. 把这些文件放进一个独立目录。
5. 统一修正 `<compiler meshdir="..." texturedir="...">` 和各级 `include` 相对路径。
6. 在目标代码库里只加载新的顶层入口 `xml`。

#### 推荐的导出目录结构

```text
my_model_bundle/
  model.xml
  assets/
    myolegs_assets.xml
    myolegs_chain.xml
    myolegs_tendon.xml
    myolegs_muscle.xml
  meshes/
  textures/
```

如果目标代码库也是 MuJoCo 生态，这是最通用的结构。

### 6.2 路径 B：你是通过项目代码自动生成 XML

项目里已经有现成导出接口：

- `src/utils/smpl_skeleton/skeleton_local.py` 里有 `write_xml(...)` 和 `write_str(...)`
- `src/utils/smpl_skeleton/skeleton_mesh_local.py` 里也有 `write_xml(...)`
- `src/utils/smpl_skeleton/smpl_local_robot.py` 里有 `write_xml(...)` 和 `export_xml_string()`

这意味着如果你的模型来源于 SMPL 骨架、或你在 `SMPL_Robot` / `Skeleton` 这条链路里生成了新结构，那么可以直接：

1. 在代码里构建模型树。
2. 调用 `write_xml("xxx.xml")` 落盘。
3. 或调用 `export_xml_string()` 拿到字符串，再写入其他系统。

### 6.3 这条项目内的自动生成链路是怎么工作的

项目里的 `src/utils/convert_kit.py` 已经给了一个可参考流程：

1. 构建 `LocalRobot(robot_cfg)`。
2. `load_from_skeleton(...)` 导入 SMPL 骨架信息。
3. `write_xml(...)` 生成 `data/xml/smpl_humanoid.xml`。
4. 再用 `SkeletonTree.from_mjcf(...)` 读取生成好的 MJCF。

所以如果你将来设计的是“基于 SMPL 或程序化骨架”的模型，完全可以复用这个导出套路。

### 6.4 导出到其他代码库时的 5 个关键注意点

1. `include` 相对路径一定要重新检查。
   很多 MuJoCo 项目能在原仓库里运行，换目录后就失效，原因几乎都是相对路径。
2. 不要只导出入口 XML。
   这个项目的大多数模型都不是单文件自洽的。
3. `meshdir` / `texturedir` 要跟目标仓库一致。
   否则 XML 能打开，但 mesh 和纹理会丢失。
4. 如果目标库不喜欢多层 `include`，最好提前“扁平化”。
   可以自己写脚本解析并展开 `include`，或者在 MuJoCo 编译后保存成最终 XML/MJB。
5. 如果目标库只需要运行，不需要继续编辑，可以考虑导出编译后的二进制模型。
   这种方式部署更稳，但不利于后续继续改结构。

### 6.5 我建议你的实际导出策略

如果你后面自己做了一套新模型，我建议按下面顺序导出：

1. 先导出“可编辑包”
   也就是保留 `xml + include + meshes + textures` 的完整目录。
2. 再导出“单入口版本”
   让其他代码库只需要知道一个 `model.xml`。
3. 最后按需要再准备“部署版”
   比如编译后的单体模型或固定目录结构版本。

这样最不容易把后续研究卡死。

## 7. 当前仓库里值得注意的两个现实问题

### 7.1 有两个旧配置仍指向过时路径

这两个文件里仍写着 `data/xml/myolegs.xml`：

- `cfg/run/run.yaml`
- `cfg/run/t2m.yaml`

但当前仓库里真正存在的是：

- `data/xml/legs/myolegs.xml`
- `data/xml/legs_abs/myolegs_abdomen.xml`
- `data/xml/legs_back/leg_soccer/myolegs_soccer.xml`

所以你后面如果直接复用旧配置，需要先修正 `xml_path`。

### 7.2 `legs_back/torso` 有上游遗留相对路径风险

`legs_back/torso/assets/myotorso_assets.xml`、`myotorso_chain.xml`、`myotorso_abdomen_assets.xml`、`myotorso_abdomen_chain.xml` 里仍出现了 `../../myo_sim/...` 这样的上游目录写法。

这说明：

- 它们原本来自上游 MyoSuite / myo_sim 目录布局。
- 如果你把 torso 子模型单独抽出来导出，几乎一定要先统一整理相对路径。

对 Kinesis 当前主链来说，最稳妥的做法仍然是优先从 `legs_abs` 或 `legs_back/leg_soccer` 这种完整入口去裁剪和导出，而不是直接从 `torso/` 子目录单独搬运。

## 8. 最后的实操建议

如果你现在就要开始动手，我建议直接按下面 3 步走：

1. 先以 `data/xml/legs_abs/myolegs_abdomen.xml` 为主工作入口，画出它的 `include` 树。
2. 先做一次“小改动”实验，例如修改一个腿部 `site` 或一个腹部 actuator 参数。
3. 验证流程跑通后，再把同类修改迁移到 `data/xml/legs_back/leg_soccer/myolegs_soccer.xml`。

这样最容易把“理解模型结构”和“开始做自己的模型”两件事衔接起来。
