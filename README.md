# scROT: 单细胞RNA最优传输药物响应预测

## 介绍

scROT 是一个基于变分自编码器 (VAE) 和最优传输 (OT) 的框架，用于整合 bulk RNA-seq 和单细胞 RNA-seq (scRNA-seq) 数据，以预测药物响应。该项目通过域适应技术桥接 bulk 和单细胞数据之间的差距，实现更准确的药物敏感性预测。

### 项目目的
- **整合多模态数据**：结合 bulk RNA-seq (源域) 和 scRNA-seq (目标域) 数据。
- **药物响应预测**：使用深度学习模型预测细胞对特定药物的敏感性（例如 Gefitinib、Vorinostat）。
- **域适应**：通过 OT 和可选的 MMD (最大均值差异) 匹配对齐不同域的分布。

### 主要功能
- 数据加载和预处理：支持 SMOTE 过采样、权重采样等方法处理不平衡数据。
- 模型训练：使用 VAE 编码器和解码器，支持共享或非共享结构。
- 最优传输：计算并可选保存 OT 计划，用于数据对齐。
- 预测和评估：计算 AUC、AUPR 等指标评估药物响应预测性能。
- 可解释性：可选使用 Integrated Gradients 分析基因贡献。

### 项目组件
- **MainRun.py**：主入口脚本，处理参数解析、数据读取、模型训练和评估。
- **scot/**：核心包目录。
  - **data_loader.py**：数据加载器，支持批量采样、SMOTE 等。
  - **function.py**：运行函数，包括模型训练和集成。
  - **model/**：模型定义。
    - **vae.py**：VAE 模型实现，包括编码、解码和损失计算。
    - **layer.py**、**loss.py**、**mmd.py**、**utils.py**：辅助模块。
- **drug/**：存储药物相关数据和结果（如 OT 计划、AUC/AUPR 指标）。

## 安装

### 要求
- Python 3.8+
- PyTorch
- Scanpy
- Scikit-learn
- Imbalanced-learn
- 其他依赖：numpy, pandas, scipy 等。

### 步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/your-repo/scROT.git
   cd scROT
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 用法

### 运行模型
使用 MainRun.py 训练模型。示例命令：
```bash
python MainRun.py --drug_name Gefitinib --batch_size 256 --n_epoch 1000 --data_path /path/to/data/
```

#### 参数说明
- `--drug_name`：药物名称 (e.g., Gefitinib)。
- `--batch_size`：批量大小。
- `--n_epoch`：训练轮数。
- `--lambda_recon`、`--lambda_kl` 等：损失权重。
- `--sampler`：采样方法 (smote, weight, none)。
- `--unshared_encoder`：是否使用非共享编码器 (1 为是)。
- `--optimal_transmission`：是否使用 OT (1 为是)。
- `--save_OT`：是否保存 OT 计划 (1 为是)。

### 输出
- 模型检查点保存在 `output/checkpoint/`。
- 性能指标 (AUC, AUPR) 保存为 TXT 文件在 `drug/{drug_name}/`。
- OT 计划可选保存为 CSV。

### 示例
1. 准备数据：放置 bulk 和 scRNA 数据在 `--data_path` 指定目录。
2. 运行：
   ```bash
   python MainRun.py --drug_name Vorinostat --sampler smote --optimal_transmission 1
   ```
3. 查看结果：检查生成的 AUC/AUPR 文件和潜在表示。

## 贡献
欢迎 Pull Requests！请确保代码符合项目风格。

## 许可证
MIT License（假设）。

## 引用
如果使用，请引用相关论文或仓库。

   


