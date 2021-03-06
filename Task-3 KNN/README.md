# K nearest neighbors

### 1. KNN算法的介绍
KNN是一种有监督算法，通过找到与给定测试样本距离最近的k个样本，根据这k个近邻的归属类别来确定样本的类别。

#### 1.1 KNN建立过程

* 给定测试样本，计算它与训练集中的每一个样本的距离。
* 找出距离近期的K个训练样本。作为测试样本的近邻。
* 依据这K个近邻归属的类别来确定样本的类别。

#### 1.2 类别的判定方法

* 投票决定，少数服从多数。取类别最多的为测试样本类别
* 加权投票法，依据计算得出距离的远近，对近邻的投票进行加权，距离越近则权重越大，设定权重为距离平方的倒数。

#### KNN既能做分类也能做回归，还可以用来做数据的预处理的缺失值填充。
