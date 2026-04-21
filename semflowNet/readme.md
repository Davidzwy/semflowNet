## 📁 代码模块说明
项目逻辑严密对应摘要中的各个技术环节：
- **`model.py` (架构核心):** 实现了基于 **Adaptive Gating** 的交叉注意力块、**Tabular AE** 以及集成 **mini-batch OT** 的流匹配逻辑。
- **`train_handler.py` (数据预处理):** 针对流量数据的异构属性和长尾特征，进行自适应量纲对齐与密度逆向采样权重计算。
- **`train.py` (联合训练):** 调用 LLM 提取协议语义先验，执行从隐空间预训练到流轨迹优化的两阶段训练任务。
- **`process.py` (受控生成):** 封装了 SDE 求解器与 CFG 引导逻辑，支持按需生成用于数据增强的攻击样本。
- **`analyze.py` (质量评估):** 提供多维统计分布度量（KS、Wasserstein）及特征相关性漂移分析。
    

## 🚀 快速开始

### 1. 准备环境
```
pip install torch sentence-transformers pot pandas numpy scikit-learn joblib
```

### 2. 模型训练
在 `train.py` 中配置数据集路径，启动 SemflowNet 训练：
```
python train.py
```
_训练过程会自动完成语义提取 -> AE 隐空间构建 -> 流匹配轨迹直线化。_

### 3. 高保真流量生成
运行 `process.py` 生成合成数据集，用于 IDS 模型的稳健性增强：
```
python process.py
```

### 4. 评估生成质量
对比合成数据与真实数据的分布一致性：
```
python analyze.py
```