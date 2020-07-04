import dgl
import warnings
import torch as th
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings('ignore')

# 创建DGL图对象g
g = dgl.DGLGraph()
# 添加10个节点
g.add_nodes(10)
# 添加10条边，且让每条边都是从对应编号的节点指向0节点
src = th.tensor(list(range(1, 10)))
g.add_edges(src, 0)


# # 画图，首先设置宽和高，然后将DGL图转化为networkx图
# plt.figure(figsize=(14, 6))
# nx.draw(g.to_networkx(), with_labels=True)
# plt.show()


# 给每个点分配特征权重
# randn是随机生成标准正态分布的值，10行3列
x = th.randn(10, 3)
# g.ndata['feature']=x，初始化特征
g.ndata['x'] = x
# # 修改某个点的特征
g.nodes[0].data['x'] = th.zeros(1, 3)
print(g.nodes[:].data['x'])


# # 给每条边分配特征权重
# g.edata['w'] = th.randn(9, 2)
# # 两种访问方式：第一种：通过边的索引访问
# g.edges[1].data['w'] = th.randn(1, 2)
# # 两种访问方式：第二种：通过边两端连接的节点访问
# g.edges[1, 0].data['w'] = th.ones(1, 2)


# # 删除节点/边的特征向量
# g.ndata.pop('x')
# g.edata.pop('w')


