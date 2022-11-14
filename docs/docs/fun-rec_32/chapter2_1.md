# 经典召回模型

---
> （本学习笔记来源于DataWhale-第32期组队学习：[推荐系统](https://datawhalechina.github.io/fun-rec/#/ch02/ch2.1/ch2.1.1/usercf?id=%e5%9f%ba%e6%9c%ac%e6%80%9d%e6%83%b3)） ,
> [B站视频讲解](https://space.bilibili.com/431850986/channel/collectiondetail?sid=339597) 观看地址

**相关算法：**

- UserCF
- ItemCF
- Swing
- 矩阵分解

## 1.协同过滤算法

### 1.1 基本思想

协同过滤（Collaborative Filtering）推荐算法是最经典、最常用的推荐算法。基本思想是：

- 根据用户之前的喜好以及其他兴趣相近的用户的选择来给用户推荐物品。

  - 基于对用户历史行为数据的挖掘发现用户的喜好偏向，并预测用户可能喜好的产品进行推荐。
  - 一般是仅仅基于用户的行为数据（评价、购买、下载等）
- 目前应用比较广泛的协同过滤算法是基于领域的方法，主要有：

  - 基于用户的协同过滤算法（UserCF）: 给用户推荐和他兴趣相似的其他用户喜欢的产品。
  - 基于物品的协同过滤算法（ItemCF）: 给用户推荐和他之前喜欢的物品相似的物品。

不管是UserCF还是ItemCF算法，重点是计算用户之间（或物品之间）的相似度。

### 1.2 相似性度量方法

**1). 杰卡德（Jaccard）相似系数**

`Jaccard`系数是衡量两个集合的相似度的一种指标，计算公式如下：

![image.png](./assets/image.png)

- 其中$N(u),N(v)$分别表示用户$u$和用户$u$交互物品的集合。
- 对于用户$u$和$u$，该公式反映了两个交互物品交集的数量占这两个用户交互物品并集的数量的比例。

用户杰卡德相似系数一般无法反映具体用户的评分喜好信息，所以常用来评估用户是否对某物品进行打分，而不是预估用户对某物品打多少分。

**2). 余弦相似度**

`余弦相似度`衡量了两个向量的夹角，夹角越小越相似。余弦相似度的计算如下，其与`杰卡德(Jaccard)`相似系数只是在分母上存在差异：

![image.png](./assets/1665557679443-image.png)

从向量的角度进行描述，令矩阵$A$为用户-物品交互矩阵，矩阵的行表示用户，列表示物品。
- 设用户和物品数量分别为$m,n$,交互矩阵$A$就是一个$m$行$n$列的矩阵。
- 矩阵中的元素均为0/1。若用户$i$对物品$j$存在交互，那么$A_{ij} = 1$,否则为0.
- 那么，用户之间的相似度可以表示为：
$$
   sim_{uv} = cos(u,v) = \frac{u.v}{|u|.|v|}
$$

向量$u,v$在形式都是one-hot类型，$u.v$表示向量点积

## 参考资料
[基于用户的协同过滤来构建推荐系统](https://mp.weixin.qq.com/s/ZtnaQrVIpVOPJpqMdLWOcw)