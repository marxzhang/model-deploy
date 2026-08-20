# 模型部署实战：剪枝 · ONNX 转换 · 量化 · TensorRT

本项目以 **ResNet34 + 花朵分类（flower_photos，5 类）** 为示例，完整走一遍深度模型从训练、剪枝、导出 ONNX、INT8 量化，到最终部署到 TensorRT 的流程。本 README 不仅告诉你**怎么跑**，还讲清楚每一步背后的**原理**，方便学习与二次开发。

---

## 目录

1. [项目结构](#项目结构)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [完整流水线总览](#完整流水线总览)
5. [第一步：训练模型](#第一步训练模型)
6. [第二步：模型剪枝](#第二步模型剪枝)
7. [第三步：单张图片推理验证](#第三步单张图片推理验证)
8. [第四步：PyTorch → ONNX](#第四步pytorch--onnx)
9. [第五步：量化（PTQ / QAT）](#第五步量化ptq--qat)
10. [第六步：ONNX → TensorRT 引擎](#第六步onnx--tensorrt-引擎)
11. [第七步：ONNX 与 TRT 结果对比](#第七步onnx-与-trt-结果对比)
12. [原理详解](#原理详解)
13. [常见问题 FAQ](#常见问题-faq)

---

## 项目结构

```
deploying_service/
├── pruning_model_pytorch/               # 剪枝专题
│   ├── model.py                         # 自实现 ResNet34 / ResNet101（键名与 torchvision 一致）
│   ├── train.py                         # 训练花朵分类模型，输出 resNet34.pth
│   ├── main.py                          # 全局非结构化剪枝（L1），统计稀疏度并验证精度
│   ├── predict.py                       # 用训练好的权重对单张图片推理
│   ├── split_data.py                    # 把 flower_photos 平铺数据切分成 train/val
│   └── resnet34-pre.pth                 # ImageNet 预训练权重（fc 为 1000 类）
├── deploying_pytorch/
│   ├── convert_onnx_cls/                # ONNX 转换专题
│   │   ├── model.py                     # 自实现 ResNet
│   │   └── main.py                      # torch.onnx.export + onnxruntime 验证
│   ├── convert_tensorrt/convert_resnet34/  # TensorRT + 量化专题
│   │   ├── my_dataset.py                # 自定义 Dataset
│   │   ├── utils.py                     # 数据划分 / 训练 / 评估函数
│   │   ├── convert_pytorch2onnx.py      # 非量化 ONNX 导出
│   │   ├── quantization.py              # PTQ 校准 + QAT 微调 + 导出 QDQ ONNX
│   │   ├── build_trt_engine.py          # ONNX → TRT INT8 引擎
│   │   └── compare_onnx_and_trt.py      # ONNX 与 TRT 输出对比
│   └── convert_openvino/                # OpenVINO 专题（本项目暂不展开）
└── data/                                # 数据放这里（或 ~/code/data）
```

> **关键约定**：`pruning_model_pytorch/model.py` 与 `convert_onnx_cls/model.py` 的 state_dict 键名和 torchvision 官方 `resnet34` **完全一致**，因此 5 类权重可以在「自实现模型 ↔ torchvision 模型」之间互相加载。这是本流水线能串起来的根基。

---

## 环境准备

推荐使用 conda 环境（本项目用 `tensorrt` 环境，Python 3.9）。

```bash
# 基础包
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime numpy pillow tqdm matplotlib absl-py pycuda==2024.1.2
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com   # 若该源不可达，见 FAQ

# TensorRT 8.6 库（系统级）
# 本机路径：/usr/local/TensorRT-8.6/lib
```

**两个必知的环境坑**（本仓库已踩过并修复）：

1. **cuDNN 与 Ada 显卡（RTX 40 系）**：torch cu118 自带 `nvidia-cudnn-cu11==8.7.0.84`，在 RTX 40 系列上卷积会**段错误崩溃**。需要升级：

   ```bash
   pip install nvidia-cudnn-cu11==8.9.6.50
   ```

   升级后 cuDNN 仍为 8.x，与 TensorRT 8.6 兼容。

2. **TensorRT builder 缺库**：pip 安装的 `tensorrt` Python 包不包含 `libnvinfer_builder_resource.so`，构建引擎前需：

   ```bash
   export LD_LIBRARY_PATH=/usr/local/TensorRT-8.6/lib:$LD_LIBRARY_PATH
   ```

---

## 数据准备

原始数据集 `flower_photos` 平铺存放 5 个类别目录（`daisy/dandelion/roses/sunflowers/tulips`，共 3670 张）。训练/评估脚本需要 `train/` 与 `val/` 结构，因此先用 `split_data.py` 切分（80% / 20%）：

```bash
cd pruning_model_pytorch
python split_data.py        # 产出 ~/code/data/flower_data/{train,val}
```

切分结果（随机种子固定，可复现）：

- train: 2939 张
- val: 731 张

同时会生成 `class_indices.json`（类别名 → 索引），供 `predict.py` 反查类别。

---

## 完整流水线总览

```
                resnet34-pre.pth (ImageNet 1000类预训练)
                          │
                          ▼
   train.py ──────► resNet34.pth (花朵 5 类) ────────────────┐
                          │                                  │
                          ▼                                  ▼
   main.py (剪枝)   pruning → 稀疏模型 + 精度报告       convert_onnx_cls/main.py
                                                          │
                                                          ▼
   quantization.py (PTQ校准 + QAT) ──► quant_model_calibrated.pth
                                          │
                                          ▼
                                   resnet34.onnx (含 QDQ 量化节点)
                                          │
                                          ▼
   build_trt_engine.py ──► trt_output/resnet34.trt (INT8)
                                          │
                                          ▼
   compare_onnx_and_trt.py  ONNX vs TRT 输出对比
```

---

## 第一步：训练模型

**运行：**

```bash
cd pruning_model_pytorch
python train.py
```

**代码做了什么**（`train.py`）：

1. 用 `resnet34()` 创建 **1000 类** 网络，以 `strict=False` 加载 ImageNet 预训练权重 `resnet34-pre.pth`。`strict=False` 允许键不严格匹配——预训练模型的 `fc`（1000 类）会被保留，随后被替换。
2. 替换分类头：`net.fc = nn.Linear(inchannel, 5)`。**只有全连接层重新初始化**，前面的卷积层全部继承 ImageNet 特征，这就是"迁移学习 / 微调"。
3. Adam 优化器 + CrossEntropyLoss，训练 3 个 epoch。
4. 每个 epoch 在验证集评估，**只有验证精度创新高才保存** `resNet34.pth`。

**原理要点**：

- **迁移学习为什么有效**：ImageNet 训练出的浅层卷积已学会"边缘 → 纹理 → 局部形状"的通用视觉特征，花朵数据量小也能直接复用这些特征，只需重新学习最后一层的语义映射。训练 3 个 epoch 精度即可到 0.93 左右。
- **`load_state_dict` 的 strict 参数**：`strict=True` 要求键完全一一对应，多一个少一个都报错；`strict=False` 会返回 `missing_keys` 与 `unexpected_keys` 供你检查，适合"替换了某些层"的场景。

**验证权重格式**（确保能被 torchvision resnet34 加载）：

```python
import torch
from torchvision.models import resnet34
sd = torch.load("resNet34.pth", map_location="cpu")
m = resnet34(num_classes=5)
missing, unexpected = m.load_state_dict(sd, strict=True)
print(missing, unexpected)   # 都应为空
```

---

## 第二步：模型剪枝

**运行：**

```bash
cd pruning_model_pytorch
python main.py
```

**代码做了什么**（`main.py`）：

1. 加载训练好的 `resNet34.pth`（5 类）。
2. 遍历模型所有 `torch.nn.Conv2d` 卷积层，收集成 `parameters_to_prune` 列表（第 80-83 行）。
3. 调用 `prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.5)`：在所有卷积层的所有权重中，**全局地**按 L1 范数大小把最小的 50% 权重置零（第 86-88 行）。
4. `count_sparsity` 统计每层和全局的稀疏度（置零比例）。
5. 在验证集上评估剪枝后的精度，观察掉点情况。

**原理要点**：

- **剪枝目的**：利用神经网络的参数冗余，删除/置零不重要参数，减少模型参数量、计算量，并提高部署效率。

- **L1 / L2 范数**：

  - **L1**：各元素绝对值之和，例如 `|w1| + |w2| + ...`。在剪枝中常用 `|w|` 衡量**单个权重的重要性**，绝对值越小通常越不重要。

  - **L2**：各元素平方和再开根号，例如 `sqrt(w1² + w2² + ...)`。在结构化剪枝中可以衡量**整个 Filter/Channel 的整体重要性**，范数越小通常越不重要。

- **非结构化剪枝**：以**单个权重**为单位。例如 `global_unstructured(..., L1Unstructured, amount=0.5)`，将所有指定层的权重统一排序，按 `|w|` 剪掉最小的 50%。粒度细、精度通常较稳定，但大量零权重不一定带来实际硬件加速。

- **结构化剪枝**：以 **Filter、Channel 等完整结构单元**为单位。例如：

  prune.ln_structured(module, name="weight",amount=0.5, n=2, dim=0)

  - `amount=0.5`：剪掉 50% 的结构单元；

  - `n=2`：使用 **L2 范数**衡量结构单元的重要性；

  - `dim=0`：对于 Conv2d 的 `[Out, In, H, W]` 权重，沿第 0 维，即剪 **输出 Filter/Channel**。

  - 结构化剪枝更容易真正降低 FLOPs、实现推理加速，但通常对精度影响更明显。

- **`prune.remove()`**：只是移除 PyTorch 的 `mask + weight_orig` 剪枝重参数化，将剪枝后的权重正式保留下来，**不会真正删除 Filter/Channel，也不会改变 Tensor shape**。

- **真正结构压缩**：需要重构网络，例如 `Conv1: 3→64` 剪掉 32 个输出 Channel 后，变成 `3→32`，同时下一层的输入 Channel 也要同步修改。可以手动完成，也可以使用 **Torch-Pruning / DepGraph** 自动分析层间依赖。

- **剪枝后通常需要 Fine-tuning**：通过重新训练让模型适应被删除的参数，恢复部分精度。

---

## 第三步：单张图片推理验证

**运行：**

```bash
cd pruning_model_pytorch
python predict.py       # 读取 ../tulip.jpg（需自己准备一张花朵图片）
```

流程：读图 → `Resize(256)+CenterCrop(224)+Normalize` → `unsqueeze(0)` 加 batch 维 → 模型前向 → `softmax` 得到概率 → 用 `class_indices.json` 反查类别名。

> 注意：`predict.py` 里图片路径是写死的 `../tulip.jpg`，运行前请放置一张花朵图片或修改该行。

---

## 第四步：PyTorch → ONNX

ONNX（Open Neural Network Exchange）是**模型中间表示（IR）**标准，让模型可以脱离 PyTorch 运行：由 ONNX Runtime、TensorRT、OpenVINO 等推理引擎加载。

**方式一：`convert_onnx_cls/main.py`（带动态 batch、含推理演示）**

```bash
cd deploying_pytorch/convert_onnx_cls
# 需要先把权重放进来：cp ../../pruning_model_pytorch/resNet34.pth .
python main.py           # 产出 resnet34.onnx + 打印单图预测
```

关键参数（`main.py`）：

- `opset_version=10`：ONNX 算子集版本。越高支持的算子越新，但推理引擎要跟上；TensorRT 8.6 建议 13+。
- `dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}`：把 batch 维声明为**动态**，导出的模型可接受任意 batch 大小。
- `do_constant_folding=True`：导出时做常量折叠，把 BN、Conv 等可合并的静态计算提前算好。

**方式二：`convert_tensorrt/convert_resnet34/convert_pytorch2onnx.py`（静态 batch，供 TRT 对比用）**

```bash
cd deploying_pytorch/convert_tensorrt/convert_resnet34
cp ../../../pruning_model_pytorch/resNet34.pth "resNet34(flower).pth"
python convert_pytorch2onnx.py    # 产出 resnet34.onnx（batch 固定为 1）
```

**原理要点**：

- **导出本质**：`torch.onnx.export` 用"假输入"（如 `torch.rand(1,3,224,224)`）对模型做一次**跟踪（trace）**，把实际执行的计算图翻译成 ONNX 算子图（Conv → BatchNormalization → ReLU …），并内嵌权重。
- **为什么需要假输入**：ONNX 是数据流图，必须有确定的输入形状才能把 `nn.Module` 的 `forward` 逻辑"实例化"成图。
- **验证**：导出后 `onnx.checker.check_model` 检查图结构合法性；再用 `onnxruntime` 跑同样的输入，和 PyTorch 输出用 `np.testing.assert_allclose` 对比，确认**数值一致性**（这就是常说的 `rtol/atol` 容差测试）。

---

## 第五步：量化（PTQ / QAT）

**运行：**

```bash
cd deploying_pytorch/convert_tensorrt/convert_resnet34
python quantization.py \
    --weights ../../../pruning_model_pytorch/resNet34.pth \
    --data-path ~/code/data/flower_photos
```

产出：

- `quant_model_calibrated.pth`：校准/微调后的量化模型权重
- `resnet34.onnx`：**含 QDQ 节点**的 INT8 量化 ONNX

**脚本流程**（`quantization.py`）：

1. `quant_modules.initialize()`：把模型里的 `Conv2d/Linear` 等模块**替换**成对应的量化版本 `QuantConv2d/QuantLinear`，这些模块内部插入了 `TensorQuantizer`（负责记录输入/权重范围并做 fake 量化）。
2. 设置激活层校准方法为 `histogram`（直方图校准），权重层为 `max`（`QuantDescriptor`）。
3. 加载训练好的 5 类权重，`model.cuda()`。
4. **PTQ 校准（Post-Training Quantization）**：`collect_stats` 在验证集上喂数据，每个 `TensorQuantizer` 收集激活值的统计信息；`compute_amax` 用 percentile=99.99 方法确定每个张量的量化范围（`amax`）。这就是"用少量数据统计出该张量应该映射到 int8 的 [-128,127] 范围"。
5. 校准后评估精度，保存 `quant_model_calibrated.pth`。
6. **QAT（Quantization-Aware Training，默认开启）**：用 SGD + 余弦学习率在训练集上微调几个 epoch，让网络"适应"量化噪声，弥补 PTQ 掉点。
7. `export_onnx`：把 fake-quant 节点以 **QDQ 模式**导出为 ONNX（`QuantizeLinear` + `DequantizeLinear` 成对出现），TensorRT 等引擎可直接消费。

**原理要点：**

- **量化（Quantization）**：用低精度表示高精度数据，如 `FP32 → INT8`，降低模型显存/存储占用，并可能提升推理速度。

- **量化基本公式**：`quant(x) = round(clamp(x / scale, -128, 127))`，`dequant(x) = x * scale`。核心是确定每个张量的 `scale = amax / 127`（权重）或 `amax / 127`（激活）。`amax` 就是该张量激活/权重的绝对最大值估计。

- **PTQ（训练后量化）**：模型训练完成后再量化。通过 **Calibration（校准）**统计 Activation 分布，确定 `scale / zero-point / amax` 等量化规则，然后转换为 INT8。

- **Calibration**：不训练模型，只用少量代表性数据跑模型，统计 Activation 的分布。

- **Histogram Calibration**：记录 Activation 的直方图，再通过 `Max / Percentile / MSE / Entropy` 等方法确定量化范围。

- **QAT（量化感知训练）**：训练时模拟 INT8 的量化误差，使模型主动适应量化，通常比直接 PTQ 精度更好。

- **Fake Quantization**：`FP32 → Quantize → INT8整数 → Dequantize → FP32近似值`；前向使用这个“伪量化值”，而不是原始 FP32 值。

- **QAT 的关键**：**前向模拟 INT8，反向更新 FP32 参数**。

- **STE（Straight-Through Estimator）**：量化/`round()` 本身难以求梯度，因此反向传播时用近似梯度，让 FP32 参数仍能通过梯度下降更新。

- **PTQ + QAT**：PTQ 负责**确定量化规则**，QAT 负责**让模型适应这套规则**，最终再将训练好的参数真正转换为 INT8。

- **PTQ 与 QAT 的区别**：

  - PTQ：不训练，只用一小批校准数据统计范围，快，但精度损失略大。
  - QAT：把量化误差（round 噪声）模拟进前向传播（fake quant），训练时梯度照样回传，让权重适应量化后的分布，精度更好。

- **校准方法**：

  - `max`：直接用张量绝对最大值，简单但对离群点敏感。
  - `histogram`（如 percentiles）：统计激活值直方图，取 99.99 分位数，能避开极端离群值，通常更稳。

- **QDQ 结构**：`x → QuantizeLinear(x, scale, zp) → DequantizeLinear → 算子`。QDQ 是 ONNX 标准的 INT8 表达，推理引擎看到"先量化再反量化"即可用真实 INT8 内核替换这段算子。

**一句话：**

> **PTQ 是“确定怎么量化”，QAT 是“让模型学会适应这种量化”。**

---

## 第六步：ONNX → TensorRT 引擎

**运行：**

```bash
cd deploying_pytorch/convert_tensorrt/convert_resnet34
export LD_LIBRARY_PATH=/usr/local/TensorRT-8.6/lib:$LD_LIBRARY_PATH
python build_trt_engine.py            # 读取 resnet34.onnx → trt_output/resnet34.trt
```

**代码做了什么**（`build_trt_engine.py`）：

1. `builder` 创建网络定义（`EXPLICIT_BATCH` 显式 batch 模式）。
2. `OnnxParser` 解析 QDQ ONNX，构建网络。
3. 配置 `BuilderConfig`：设置 workspace 内存上限、开启 `INT8` flag。
4. 添加 `OptimizationProfile`：给动态输入 `input` 设置 `min/opt/max` 形状（本项目都是 `(1,3,224,224)`）。
5. `builder.build_serialized_network` 完成**引擎构建（图优化 + kernel 选择）**，序列化保存。

**原理要点**：

- **TensorRT 构建是离线优化**：它把 ONNX 图做算子融合（如 Conv+Bias+ReLU 合并）、层重排、选择对应 GPU 的最优 INT8 kernel，生成一个**绑定硬件/精度配置的引擎**（.trt 文件），运行时代价最小。
- **为什么 QDQ 模型构建 INT8 不需要校准数据**：量化范围（scale）已经写死在 ONNX 的 QDQ 节点里，TensorRT 直接用，无需像传统"FP32 模型 + calibrator"那样重新收集统计。这就是第五步量化产出的 QDQ 模型的价值。
- **警告解读**：构建时可能出现 "37 weights outside int8 range, clipped to int8 range"——说明个别权重超出 int8 范围被截断，属正常现象，会带来轻微精度影响。

---

## 第七步：ONNX 与 TRT 结果对比

**运行：**

```bash
cd deploying_pytorch/convert_tensorrt/convert_resnet34
export LD_LIBRARY_PATH=/usr/local/TensorRT-8.6/lib:$LD_LIBRARY_PATH
python compare_onnx_and_trt.py
```

`compare_onnx_and_trt.py` 用同一张随机图分别跑 onnxruntime（ONNX 模拟量化）和 TensorRT（真 INT8 内核），对比输出 logits：

- 用 `np.testing.assert_allclose(..., rtol=0.1, atol=0.02)` 校验数值接近（INT8 量化误差级别，比 FP32 的 1e-5 宽）。
- 打印两边的 argmax 预测类别，确认**分类结果一致**。

**原理要点**：

- onnxruntime 在 CPU 上执行 QDQ 节点 = "先量化再反量化"，是**模拟量化**；TensorRT 用 GPU 上的真实 INT8 算子，两者计算路径不同，logits 有微小差异（本项目最大约 0.005）属正常。真正要保证的是**预测类别一致**。

---

## 原理详解

### 1. 剪枝（Pruning）

- **目标**：删除模型中不重要的参数，降低存储与计算量。两类方法：
  - **非结构化剪枝**：逐个权重置零（本项目 `global_unstructured` + `L1Unstructured`）。优点：简单、压缩率高；缺点：权重稀疏但形状不变，通用硬件难以加速。
  - **结构化剪枝**：整行/整卷积核删除（代码注释示例 `ln_structured(..., dim=0)`）。优点：真删结构，利于加速；缺点：需要保证网络连通性、通常需要重训练恢复。
- **判定"不重要"**：L1 范数（绝对值之和）小 = 该权重/卷积核对输出贡献小。
- **torch 实现机制**：剪枝不真正改权值，而是生成 `weight_mask` 掩码，前向时 `weight = weight_orig * weight_mask`。`prune.remove` 才把 mask 永久固化进权重。
- **剪后必做**：微调（fine-tune）恢复精度，或与量化组合使用。

### 2. ONNX 转换

- ONNX = 模型交换格式（计算图 + 权重 + 元信息），与框架无关。
- `torch.onnx.export` 通过 **tracing**（跟踪实际执行路径）把 PyTorch 算子映射到 ONNX 算子。
- 关键概念：`opset_version`（算子集版本）、`dynamic_axes`（动态维度）、`do_constant_folding`（常量折叠，如融合 BN）。
- 验证闭环：`onnx.checker`（图合法性）→ `onnxruntime` 推理 → 与 PyTorch 输出对比容差。

### 3. 量化（Quantization）

- **统一公式**：INT8 表示 `[-128, 127]`，张量实际范围 `[-amax, amax]`，`scale = amax / 127`。量化 = 除以 scale 并取整；反量化 = 乘以 scale。误差来自 round 的舍入（量化噪声）。
- **PTQ**：不训练。用校准数据统计每个激活张量的 `amax`。校准方法：`max`（易受离群点影响）vs `histogram` + percentile（更鲁棒）。
- **QAT**：训练时用"fake quant"（前向走量化+反量化，反向正常 BP）模拟 INT8 行为，让权重适应量化误差，精度损失更小。
- **QDQ 表达**：ONNX 用 `QuantizeLinear / DequantizeLinear` 对表达量化区间，推理引擎据此自动使用 INT8 内核，无需重复校准。

### 4. TensorRT 部署

- **构建（build）**：离线完成图优化（算子融合、层重排）、精度策略选择（FP32/FP16/INT8）、kernel 挑选，产出序列化引擎。
- **运行（runtime）**：反序列化引擎，用 CUDA 流异步推理，host↔device 数据搬运用 PyCUDA。
- **动态 shape**：通过 OptimizationProfile 声明 `min/opt/max`，运行时 `set_binding_shape` 指定实际尺寸。

### 5. 各环节精度损失来源

| 环节      | 损失来源           | 本项目中观察         |
| --------- | ------------------ | -------------------- |
| 剪枝 50%  | 权重置零           | 精度下降，需微调     |
| FP32→INT8 | round 量化噪声     | logits 差 ~0.005     |
| QAT       | 训练收敛到量化分布 | 精度可回到 ~0.98+    |
| ONNX→TRT  | 算子融合/内核差异  | 与 ONNX 预测类别一致 |

---

## 常见问题 FAQ

**Q1：`torch.onnx.export` 报 `enable_onnx_checker` 参数错误？**
torch 2.x 移除了该参数。删掉它，改用导出后 `onnx.checker.check_model(onnx_model)` 校验。本项目已如此处理。

**Q2：TensorRT builder 报 `libnvinfer_builder_resource.so` 找不到？**
pip 版 `tensorrt` 不含完整库。执行 `export LD_LIBRARY_PATH=/usr/local/TensorRT-8.6/lib:$LD_LIBRARY_PATH`。

**Q3：训练/推理段错误（Segmentation fault）？**
RTX 40 系 + torch cu118 自带 cuDNN 8.7 的已知 bug。升级：`pip install nvidia-cudnn-cu11==8.9.6.50`。

**Q4：`pytorch-quantization` 装不上 / NGC 源不可达？**
本项目环境 NGC 源被代理 TLS 阻断，改为从 NVIDIA 官方 GitHub（`v8.6.1` 标签 `tools/pytorch-quantization`）下载源码本地构建安装：

```bash
pip install ./pqsrc --no-deps --no-build-isolation
```

（需要 torch 的 CUDA 扩展编译环境：nvcc + gcc）

**Q5：权重文件名对不上？**
`convert_pytorch2onnx.py` 期望 `resNet34(flower).pth`，`quantization.py` 默认 `resNet(flower).pth`。统一做法：用 `--weights` 参数直接指向 `resNet34.pth`（quantization 已支持），或 `ln -s resNet34.pth resNet34(flower).pth`。

**Q6：onnxruntime 与 TRT 对比差很多？**
确认对比的是"模拟量化（ONNX QDQ）vs 真 INT8（TRT）"，容差应按 INT8 级别设置（atol≈0.02、rtol≈0.1），并重点看 argmax 类别是否一致。

**Q7：剪枝后模型大小没变小？**
非结构化剪枝置零但形状不变，文件大小基本不变（除非用稀疏存储）。想真正减小体积/加速请用结构化剪枝或量化（INT8 体积为 1/4）。

---

## 部署阶段的优化

模型推理速度
├── ① 模型层面
│   ├── 剪枝
│   ├── 量化
│   ├── 换轻量模型/减少输入分辨率
│   └── 算子/结构优化
│
├── ② TensorRT 编译层面
│   ├── Layer Fusion
│   ├── Tensor Core
│   ├── 最优 tactic
│   └── Q/DQ Fusion
│
├── ③ 推理执行层面
│   ├── Batch
│   ├── CUDA Graph
│   ├── Multi-Stream
│   └── Buffer 复用
│
└── ④ 工程层面
    ├── 减少 CPU↔GPU 拷贝
    ├── 异步预处理/后处理
    └── Pipeline 并行
