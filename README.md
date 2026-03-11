# 大模型Case聚类实战项目

## 项目说明
基于100条真实房产通话数据，完成文本向量化和聚类分析。

## 核心文件
- `30168868.csv` - 原始ASR数据
- `conversations_with_summary.xlsx` - 100条对话及AI生成的摘要
- `step2_embedding.py` - 向量化
- `step3_kmeans_visualization.py` - K-Means聚类
- `step4_dbscan_bad_cases.py` - DBSCAN异常检测

## 执行流程

### 1. 向量化
```bash
python step2_embedding.py
```
生成：`embeddings.npy`

### 2. K-Means聚类
```bash
python step3_kmeans_visualization.py
```
生成：`kmeans_clustering.png`

### 3. DBSCAN异常检测
```bash
python step4_dbscan_bad_cases.py
```
生成：`dbscan_bad_cases.png`

## 面试要点
- 向量化：TF-IDF，无需下载大模型
- K-Means：需指定K=3，找出质心和典型样本
- DBSCAN：自动发现簇，eps控制密度阈值
- PCA降维：768维→2维用于可视化
