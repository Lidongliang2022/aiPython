# 大模型Case聚类实战项目

## 项目说明
这是一个完整的ASR通话数据聚类分析项目，帮助你掌握：
- 大模型文本摘要
- 向量化(Embedding)
- K-Means聚类
- DBSCAN异常检测

## 环境准备

```bash
pip install -r requirements.txt
```

## 使用步骤

### 1. 提取摘要 (step1_extract_summary.py)
- 解析CSV中的ASR对话数据
- 调用大模型API生成100字摘要
- **需要配置API Key**

```python
# 修改step1中的配置
API_KEY = "your-api-key-here"
BASE_URL = "https://api.deepseek.com"  # 或智谱API
```

运行：
```bash
python step1_extract_summary.py
```

### 2. 向量化 (step2_embedding.py)
- 使用本地中文向量模型
- 将摘要转为768维向量
- 首次运行会自动下载模型

运行：
```bash
python step2_embedding.py
```

### 3. K-Means聚类可视化 (step3_kmeans_visualization.py)
- K=3聚类
- 找出每个簇的典型代表（质心最近样本）
- PCA降维到2D并绘制散点图

运行：
```bash
python step3_kmeans_visualization.py
```

**输出**: `kmeans_clustering.png`

### 4. DBSCAN异常检测 (step4_dbscan_bad_cases.py)
- 模拟20条Bad Case
- 自动发现错误簇和孤立噪点
- 可视化展示

运行：
```bash
python step4_dbscan_bad_cases.py
```

**输出**: `dbscan_bad_cases.png`

## 面试要点

### 技术细节
- **7B模型**: 70亿参数，FP16约14GB显存，量化后6-8GB
- **向量维度**: 768维（m3e/text2vec模型标准）
- **质心**: K-Means的聚类中心，代表该簇的典型特征
- **降维**: PCA将768维压缩到2维用于可视化

### 回答话术
"我用text2vec提取768维向量，通过K-Means找到质心，用PCA降维画散点图验证分布。DBSCAN用于发现未知错误模式，eps参数控制密度阈值。"

## 文件说明
- `30168868.csv`: 原始ASR数据
- `summaries.csv`: 提取的摘要
- `embeddings.npy`: 向量数据
- `*.png`: 可视化结果
