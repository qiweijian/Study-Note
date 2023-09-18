最近要学习一些知识图谱的基础知识，先顺着斯坦福CS223W的第九节开始往下看。[b站链接](https://www.bilibili.com/video/BV1RZ4y1c7Co/?p=26&share_source=copy_web&vd_source=cc8c0d05fcf9121dcabb4f4195f81f67)

## GNN

GNN有两个基本操作， 转换 Transformation 和聚合 Aggregation。
- 转换是指将一个节点的特征映射到另一个（统一的）特征空间
- 聚合是指将一个节点的邻居节点的特征汇总

几个常用的GNN模型对比如下

| Graph Neural Network | Aggregation Method         | Transformation Method |
|----------------------|----------------------------|-----------------------|
| GCN                  | Mean pool                  | Linear + ReLU         |
| GraphSAGE            | Max pool         | MLP       |
<!-- | GAT                  | Attention-based aggregation| Linear + LeakyReLU    |
| ChebNet              | Spectral convolution       | Linear                | -->

### GNN的表征能力

#### 计算图 Computational Graph




