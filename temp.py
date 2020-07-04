import dgl
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')


# 设置好dgl图
def build_karate_club_graph():
    g = dgl.DGLGraph()
    g.add_nodes(34)
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]

    # tuple()函数创建元组对象
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # 在dgl图中，边是有方向的，因此需要使其双向
    g.add_edges(dst, src)

    return g


# 输出节点数和边缘数，并画图
G = build_karate_club_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

# plt.figure(figsize=(14, 6))
nx_G = G.to_networkx().to_undirected()
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.show()

# 给边和节点赋予特征
# torch.eye()函数：返回一个2维张量，对角线位置全1，其它位置全0
G.ndata['feat'] = torch.eye(34)


# 定义一个GCN
# 主要定义message方法和reduce方法
# NOTE: 为了易于理解，整个教程忽略了归一化的步骤
def gcn_message(edges):
    # 得到计算后的batch of edges的信息，这里直接返回边的源节点的feature.
    return {'msg' : edges.src['h']}


def gcn_reduce(nodes):
    # 得到计算后batch of nodes的信息，这里返回每个节点mailbox里的msg的和
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}


# 定义GCN层
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g为图对象； inputs为节点特征矩阵
        # 设置图的节点特征
        g.ndata['h'] = inputs
        # 触发边的信息传递
        g.send(g.edges(), gcn_message)
        # 触发节点的聚合函数
        g.recv(g.nodes(), gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('h')
        # 线性变换
        return self.linear(h)


# 定义一个简单的2层GCN网络
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


# 以空手道俱乐部为例，第一层将34层的输入转化为隐层为5，第二层将隐层转化为最终的分类数2
net = GCN(34, 5, 2)

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])
labels = torch.tensor([0, 1])


# 训练过程
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    logits = net(G, inputs)
    # 将日志保存下来以便以后查看
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))