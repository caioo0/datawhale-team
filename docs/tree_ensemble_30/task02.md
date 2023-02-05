# Part A: 决策树（下）
----
（本学习笔记来源于[DataWhale-树模型与集成学习](https://datawhalechina.github.io/machine-learning-toy-cod)） 

```md
The answer must be in the attempt.
你所追寻的答案一定会在努力探索中展现。
```


本次内容：

实现`CART`的分类树算法代码


关键知识：`熵`,`信息熵` ,`条件熵`,`信息增益`,`信息增益比`,`基尼系数`，`剪枝`

## 一、基本概念知识


### 1. 信息熵：

1.1 熵：信息的期望值，表示随机变量不确定性的度量.如果待分类的事物可能划分在多个分类之中，则符号$x_{i}$的信息定义为 ：

$$
    I(x_{i})  = log_{2}\frac{1}{p(x_{i})} = - log_{2}p(x_{i})
$$

   其中，$p(x_{i})$是选择该分类的概率

1.2 信息熵：计算所有类别所有可能值包含的信息期望值(数学期望)：

$$
    H = -\sum^{n}_{i=1}p(x_{i})log_{2}p(x_{i})
$$

   其中，$p(x_{i})$是选择该分类的概率，$n$是分类的数目。熵越大，随机变量的不确定性就越大。
   

当熵中的概率由数据估计(特别是最大似然估计)得到时，所对应的熵称为经验熵(empirical entropy)，信息熵又称为经验熵。

信息熵公式


```python
"""
函数说明：计算给定数据值得信息熵（经验熵）

Parameters:
    dataSet - 数据集（假定所给数据集向量最后一列为标签（Label）信息）
Returns:
    commEnt - 信息熵（经验熵）
Author:
    Jo Choi (297495363@qq.com)
Modify:
    2021-10-16
"""

def calcomEntropy(dataSet,labelIndex):
    rowData = len(dataSet)                       # 返回数据集得行数（样本容量|D|）
    labelCounts = {}                                 # 保存每个标签（Label）出现次数的字典
    for colVec in dataSet:                          # 遍历数据集（获取每个样本标签信息(label),并保存统计数）
        currentLabel = colVec[labelIndex]                    # 提取标签（Label）信息，//rowVec[-1] 最后一列为标签
        if currentLabel not in labelCounts.keys():  # 判断当前标签（label）没有在统计次数字典，添加初始值为0
            labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1               # 标签计数
         
    commEnt = 0.0                                     # 初始化信息熵（经验熵）为 0 
    for key in labelCounts:                          # 遍历标签(labelCounts)统计字典 
        prob    = float(labelCounts[key]) / rowData  # 计算概率
        commEnt -= prob * log(prob,2)                #  遍历计算公式
    return commEnt                                   # 返回信息熵（经验熵）
```

例如1：
有10个数据，一共有两个类别，A类和B类。其中有7个数据属于A类，则该A类的概率即为十分之七。其中有3个数据属于B类，则该B类的概率即为十分之三。
我们定义贷款申请样本数据表中的数据为训练数据集D，训练数据集D的经验熵为H(D)，|D|表示其样本容量，及样本个数。设有K个类Ck, = 1,2,3,...,K,|Ck|为属于类Ck的样本个数，因此经验熵公式就可以写为 ：

$$
    H(D) = -\sum^{k}_{k=1}\frac{|C_{k}|}{|D|}log_{2}\frac{|C_{k}|}{|D|} 
$$

$\frac{|C_{k}|}{|D|}$ 如同前面公式$p(x_{i})$，代入数据得出经验熵H(D):

$$
  H(D) = -\frac{7}{10}log_{2}\frac{7}{10}-\frac{3}{10}log_{2}\frac{3}{10} = 0.88
$$
 

代码实现：


```python
from math import log

# 直接计算
H = -log(0.7,2)*0.7 - log(0.3,2)*0.3
print("直接计算信息熵:",H)

# 调用calcomEntropy函数
# 首先转换数据集，这里只用了一个特征
dataset = [['A'],['A'],['A'],['A'],['A'],['A'],['A'],['B'],['B'],['B']]
print("调用函数计算:",calcomEntropy(dataset,-1))

```

    直接计算信息熵: 0.8812908992306927
    调用函数计算: 0.8812908992306927
    

例如2：我们看一组贷款申请样本数据表：

![img](https://cuijiahua.com/wp-content/uploads/2017/11/m_2_2.jpg)

根据此公式计算经验熵H(D)，分析贷款申请样本数据表中的数据。最终分类结果只有两类，即放贷和不放贷。根据表中的数据统计可知，在15个数据中，9个数据的结果为放贷，6个数据的结果为不放贷。所以数据集D的经验熵H(D)为：

$$
  H(D) = -\frac{9}{15}log_{2}\frac{9}{15}-\frac{6}{15}log_{2}\frac{6}{15} = 0.971
$$
 

我们先转换数据格式： 
- 年龄：{0,1,2} 0:青年，1:中年，2:老年；
- 有工作：{0,1} 0:否，1:是；
- 有自己的房子：{0,1} 0:否，1:是；
- 信贷情况：{0,1,2} 0:一般，1:好，2:非常好；
- 类别(是否给贷款)：{'no','yes'} no:否，yes:是。

创建转换后数据集:


```python
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],          # 根据表格生成数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']             # 标签分类属性
    return dataSet, labels                 # 返回数据集和标签分类属性
```

计算信息熵（经验熵）


```python
# 调用calcomEntropy函数
dataset,labels = createDataSet()
print(calcomEntropy(dataset,-1))
```

    0.9709505944546686
    

### 条件熵


这里直接给出公式：

$$
H(D|A)  = -\sum^{n}_{i=1}\frac{|D_{i}|}{|D|}H(D_{i}) = -\sum^{n}_{i=1}\frac{|D_{i}|}{|D|}\sum^{k}_{k=1}\frac{|D_{ik}|}{|D_{i}|}log_{2}\frac{|D_{ik}|}{|D_{i}|} 
$$

以贷款申请样本数据表为例进行说明。

看下年龄这一列的数据，也就是特征A=年龄，一共有三个类别，分别是：`青年`、`中年`和`老年` 刚好是5个，分别占三分之一。

有： 
$$
H(D|A=年龄)  =  (\frac{|D_{青年}|}{D} × H(D_{青年})+(\frac{|D_{中年}|}{D}×H(D_{中年})+(\frac{|D_{老年}|}{D}×H(D_{老年}))
$$
$$
         =  (\frac{5}{15} × H(D_{青年})+(\frac{5}{15}×H(D_{中年})+(\frac{5}{15}×H(D_{老年}))
$$



 
 - $H(D_{青年}) = -\frac{2}{5}log_{2}\frac{2}{5}-\frac{3}{5}log_{2}\frac{3}{5}$ ,青年最终贷款的概率为`五分之二`，不放贷为`五分之三`
 
 
 - $H(D_{中年}) = -\frac{3}{5}log_{2}\frac{3}{5}-\frac{2}{5}log_{2}\frac{2}{5}$ ,青年最终贷款的概率为`五分之三`，不放贷为`五分之二`
 
 
 - $H(D_{青年}) = -\frac{4}{5}log_{2}\frac{4}{5}-\frac{3}{5}log_{2}\frac{1}{5}$ ,青年最终贷款的概率为`五分之四`，不放贷为`五分之一`
 
 
 
 那么特征=年龄条件熵得出最终数据为：
 
 $$
       H(D|A=年龄)   =  \frac{5}{15} ×(-\frac{2}{5}log_{2}\frac{2}{5}-\frac{3}{5}log_{2}\frac{3}{5})  
         + \frac{5}{15}× (-\frac{3}{5}log_{2}\frac{3}{5}-\frac{2}{5}log_{2}\frac{2}{5})  
         + \frac{5}{15}×(-\frac{4}{5}log_{2}\frac{4}{5}-\frac{3}{5}log_{2}\frac{1}{5})   = 0.888
$$


公式实现：

- 选定特征A 
- 根据选定特征获取分布{A1,A2,A3}
- 获取特征分布子数据集


```python
"""
函数说明：按照给定特征划分数据集　

Parameters:
    dataSet - 带划分的数据集
    axis    - 划分数据集的特征
    value   - 需要返回的特征的值
Returns:
    retDataSet - 划分特征子数据集
Author:
    Jo Choi (297495363@qq.com)
Modify:
    2021-10-16
"""
def splitDataSet(dataSet,axis,value):
    retDataSet = []                                 #  创建返回的数据集列表
    for colVec in dataSet:                         # 遍历数据集
        if colVec[axis] ==value:                   # 判断符合条件的
            reduceFeatVec = colVec[:axis]          # 去掉axis特征，返回[0:axis)元素数据
            reduceFeatVec.extend(colVec[axis+1:])  # 追加后面的特征数据
            retDataSet.append(reduceFeatVec)
    return retDataSet                             # 返回特征划分数据集

"""
函数说明：计算选定特征数据集的条件熵　

Parameters:
    dataSet - 所有数据集
    labelIndex - 标签索引 
Returns:
    retDataSet - 返回每个特征的条件熵
Author:
    Jo Choi (297495363@qq.com)
Modify:
    2021-10-16
"""
def conditionalEnt(dataSet,labelIndex):
    numFeatures = len(dataset[0])-1           # 特征数，最后一个为标签值
    retDataSet = []
    for i in range(numFeatures):
        featList = [rowdata[i] for rowdata in dataset]  # 获取dataSet的第i个所有特征数
        uniqueVals = set(featList)                       # 创建set集合{}，去重
        conditEnt = 0.0                                  # 初始化特征条件熵
        for value in uniqueVals:
            subdataSet= splitDataSet(dataset,i,value)    # 划分后的子集（subdataSet）
            prob = len(subdataSet)/float(len(dataset))   # 子集的概率 
            conditEnt += prob * calcomEntropy(subdataSet,labelIndex)# 根据公式计算条件熵 子集占所有数据集概率 叉× 子集信息熵 （叠加）
        retDataSet.append(conditEnt)

    return retDataSet                             # 返回特征划分数据集

if __name__ == '__main__':
    dataset,labels = createDataSet()               # 获取数据集
    # 获取条件熵
    conditEntVec = conditionalEnt(dataset,-1)      # 获取条件熵 
    for i,value in (enumerate(conditEntVec)):
        print("第%d个特征条件熵为%.3f" % (i, value))    
        print("----")
```

    {'no': 3, 'yes': 2}
    {'no': 2, 'yes': 3}
    {'yes': 4, 'no': 1}
    {'no': 6, 'yes': 4}
    {'yes': 5}
    {'no': 6, 'yes': 3}
    {'yes': 6}
    {'no': 4, 'yes': 1}
    {'no': 2, 'yes': 4}
    {'yes': 4}
    第0个特征条件熵为0.888
    ----
    第1个特征条件熵为0.647
    ----
    第2个特征条件熵为0.551
    ----
    第3个特征条件熵为0.608
    ----
    

### 信息增益 


特征𝑨对训练数据集D的信息增益𝒈(𝑫, 𝑨)，定义为集合𝑫的经验熵𝑯(𝑫)与特征𝑨给定条件下𝑫的经验条件熵𝑯(𝑫|𝑨)之差，即：

$$
g(D,A)  = H(D) - H(D|A)
$$

在特征选择的过程中，需要选择信息增益值最大的的特征𝑨。


**算法 5.1 （信息增益的算法）**

输入：训练数据集 D 和 特征 A;  
输出：特征 A 对训练数据集 D 的信息增益 $g(D,A)$。  

（1）计算数据集 D 的经验熵（信息熵）$H(D)$   
（2）计算特征 A 对数据集 D 的经验条件熵（条件熵） ）$H(D|A)$  
（3）计算信息增益


以贷款申请样本数据表为例,计算每个特征的信息增益：



```python
def calinfoGain(dataset):
    commEntVec = calcomEntropy(dataset,-1)       # 计算信息熵
    conditEntVec = conditionalEnt(dataset,-1)    # 计算条件熵
    for i,value in enumerate(conditEntVec):   # 遍历特征的条件熵
        infoGain = commEntVec - value         # 计算信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))    
        print("----")
 
if __name__ == '__main__':
    
    dataset,labels = createDataSet()
    calinfoGain(dataset)
    
```

    第0个特征的增益为0.083
    ----
    第1个特征的增益为0.324
    ----
    第2个特征的增益为0.420
    ----
    第3个特征的增益为0.363
    ----
    

### 信息增益比

信息增益比算法修正了信息增益算法中会对某一特征取值较多时产生偏向的情况。

 信息增益比 = 惩罚参数 * 信息增益
 
 公式：
 
 $$
    g_{R}(D,A)  = \frac{g(D,A)}{H_{A}(D)}
 $$
 
 其中：
$$
 H_{A}(D) = - \sum^{n}_{i=1}\frac{|D_{i}|}{|D|}log_{2}\frac{|D_{i}|}{|D|},n 是特征 A 取值的个数
 $$
 

 
 以贷款申请样本数据表为例,计算特征{年龄}的信息增益比( 青年=5 ， 中年=5 ， 老年=5)：
 
$$
 H_{A}(D) = - (\frac{5}{15}log_{2}\frac{5}{15} + \frac{5}{15}log_{2}\frac{5}{15} + \frac{5}{15}log_{2}\frac{5}{15})
 $$
 
 


```python

```

    直接计算信息熵: 1.5897956906195279
    


```python

"""
函数说明：计算给定数据值得信息熵（经验熵）

Parameters:
    dataSet - 数据集（假定所给数据集向量最后一列为标签（Label）信息）
Returns:
    commEnt - 信息熵（经验熵）
Author:
    Jo Choi (297495363@qq.com)
Modify:
    2021-10-16
"""

def calcomEntropy(dataSet,labelIndex):
    rowData = len(dataSet)                       # 返回数据集得行数（样本容量|D|）
    labelCounts = {}                                 # 保存每个标签（Label）出现次数的字典
    for colVec in dataSet:                          # 遍历数据集（获取每个样本标签信息(label),并保存统计数）
        currentLabel = colVec[labelIndex]                    # 提取标签（Label）信息，//rowVec[-1] 最后一列为标签
        if currentLabel not in labelCounts.keys():  # 判断当前标签（label）没有在统计次数字典，添加初始值为0
            labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1               # 标签计数
         
    commEnt = 0.0                                     # 初始化信息熵（经验熵）为 0 
    for key in labelCounts:                          # 遍历标签(labelCounts)统计字典 
        prob    = float(labelCounts[key]) / rowData  # 计算概率
        commEnt -= prob * log(prob,2)                #  遍历计算公式
    return commEnt                                   # 返回信息熵（经验熵）

"""
函数说明：计算信息增益比

Parameters:
    dataSet - 数据集
    labelindex - 标签索引
Returns:
    无
Author:
    Jo Choi (297495363@qq.com)
Modify:
    2021-10-16
"""

def calinfoGainRadio(dataset,labelindex):

    commEnt = calcomEntropy(dataset,-1)          # 计算信息熵
    conditEntVec = conditionalEnt(dataset,-1)    # 计算条件熵
    
    for i,value in enumerate(conditEntVec):      # 遍历特征的条件熵
        fealEnt = calcomEntropy(dataset,i)       # 计算惩罚系数（特征熵)
        infoGain = commEnt - value               # 计算信息增益
        
        infoGainRadio = float(infoGain)/float(fealEnt)
        

        print("第%d个特征的特征熵%.3f，信息增益%.3f，增益比%.3f" % (i,fealEnt,infoGain,infoGainRadio))  
        print("----")
 
if __name__ == '__main__':
    
    dataset,labels = createDataSet()
    calinfoGainRadio(dataset,-1)
```

    第0个特征的特征熵1.585，信息增益0.083，增益比0.052
    ----
    第1个特征的特征熵0.918，信息增益0.324，增益比0.352
    ----
    第2个特征的特征熵0.971，信息增益0.420，增益比0.433
    ----
    第3个特征的特征熵1.566，信息增益0.363，增益比0.232
    ----
    

## 二、ID3 算法生成决策树

ID3 算法的核心是在决策树各个结点上应用信息增益选择特征，递归地构建决策树。

具体方法是：从根结点（root node）开始，对结点计算所有可能的特征的信息增益，选择`信息增益最大`的特征作为结点，由该特征的不同取值建立子结点；
再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增益均很小或没有特征可以选择为止。最后得到一棵决策树。

ID3 相当于极大似然法进行概率模型的选择。

  2.1 算法（ID3 算法）
 
  输入： 训练数据集D,特征值A 阈值 $\varepsilon$；  
  输出： 决策树T。    
  （1） 若 D 中所有实例属于同一类$C_{k}$,则 T 为单结点树，并将类 $C_{k}$ 作为该结点的类标记，返回 T；   
  （2） 若 $A = \phi $，则 T 为单结点树，并将 D 中实例数最大的类 $C_{k}$ 作为该结点的类标记，返回 T；    
  （3）否则，计算信息增益，选择信息增益最大的特征 $A_{g}$；   
  （4）如果 $A_{g}$ 的信息增益小于阈值  $\varepsilon$ ，则 T 为单结点树，并将 D 中实例树最大的类 $C_{k}$ 作为该结点的类标记，返回 T；  
  （5）否则，对 $A_{g}$ 的每一可能值 $a_{i}$，依 $A_{g} = a_{i}$ 将 D 分割为若干非空子集 $D_{i}$， 将 $D_{i}$ 中实例数最大的类作为标记，构建子结点，由结点及其子结点构成树T，返回T；    
  （6）对第 $i$ 个子结点，以 $D_{i}$ 为训练集，以 $ A - {A_{g}}$ 为特征集，递归地调用上面步骤，得到子树$T_{i}$,返回T；   
  
  






```python

"""
函数说明:创建决策树ID3 
 
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
Author:
   Jo Choi
Modify:
    2021-10-17
"""
```




    '\n函数说明:创建决策树ID3 \n \nParameters:\n    dataSet - 训练数据集\n    labels - 分类属性标签\n    featLabels - 存储选择的最优特征标签\nReturns:\n    myTree - 决策树\nAuthor:\n    Jack Cui\nBlog:\n    http://blog.csdn.net/c406495762\nModify:\n    2017-07-25\n'



实例：https://cloud.tencent.com/developer/article/1648827


```python

```
