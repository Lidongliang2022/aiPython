# 大模型Case聚类实战项目

## 项目说明
基于100条真实房产通话数据，完成文本向量化和聚类分析。

## 目录结构
```text
aiPython/
├── data/               # 原始数据与中间产物 (CSV, XLSX, NPY)
├── scripts/            # 核心业务逻辑脚本
├── output/             # 可视化图表与分析报告
├── README.md           # 项目说明
└── requirements.txt    # 环境依赖
```

## 核心文件说明
- `data/30168868.csv` - 原始ASR数据
- `data/conversations_with_summary.xlsx` - 100条对话及AI生成的摘要
- `scripts/step2_embedding.py` - 向量化脚本
- `scripts/step3_kmeans_visualization.py` - K-Means聚类与可视化
- `scripts/step4_dbscan_bad_cases.py` - DBSCAN异常检测
- `scripts/generate_analysis_report.py` - 生成深度分析报告

## 执行流程

> **注意**：由于脚本已整理至 `scripts/` 目录，执行时请在根目录下使用以下命令：

### 1. 向量化
```bash
python scripts/step2_embedding.py
```
生成：`data/embeddings.npy`

### 2. K-Means聚类
```bash
python scripts/step3_kmeans_visualization.py
```
生成：`output/kmeans_clustering.png`

### 3. DBSCAN异常检测
```bash
python scripts/step4_dbscan_bad_cases.py
```
生成：`output/dbscan_bad_cases.png`

### 4. 生成分析报告
```bash
python scripts/generate_analysis_report.py
```
生成：`output/cluster_analysis_report.md`

## 面试要点
- **向量化**：使用 BGE-Large 中文语义模型，提取 768 维稠密向量。
- **K-Means**：通过质心距离识别典型场景，利用边界分析识别模糊场景。
- **DBSCAN**：自动发现孤立点（噪点），用于挖掘业务中的 Bad Case。
- **PCA降维**：将高维向量投影至 2 维空间进行业务可视化。
