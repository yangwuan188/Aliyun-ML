# 基于逻辑回归的分类预测

### 1. 逻辑回归的介绍

逻辑回归实际上是一个分类模型，其优点为 ***模型简单*** 和 ***可解释性强***。 其缺点是由于模型复杂度不高，容易造成模型欠拟合的现象。
逻辑回归模型现在同样是很多分类算法的基础组件,比如分类任务中基于GBDT算法+LR逻辑回归实现的信用卡交易反欺诈，CTR(点击通过率)预估等，其好处在于输出值自然地落在0到1之间，并且有概率意义。模型清晰，有对应的概率学理论基础。它拟合出来的参数就代表了每一个对结果的影响。也是一个理解数据的好工具。但同时由于其本质上是一个线性的分类器，所以不能应对较为复杂的数据情况。
### 2.逻辑回归算法实现

借助sklearn机器学习库可轻松实现逻辑回归函数的构建：

***from sklearn.linear_model import LogisticRegression***                           （导入逻辑回归模型函数）

***lr_clf = LogisticRegression()***                                                                      （调用逻辑回归函数）
        
***lr_clf = lr_clf.fit(x_fearures, y_label)***                                                           （用逻辑回归模型拟合构造的数据集）

以上可直接用已有数据训练出一个逻辑回归模型。接着用 predict() 方法可对新数据进行预测：

***y_predict = lr_clf.predict(x_fearures_new)***                                                   (用拟合的模型来预测新数据)

***y_predic_proba = tlr_clf.predict_proba(x_fearures_new)***                          （输出预测分类的概率）
### 3. 基于鸢尾花（iris）数据集的逻辑回归分类实践

本次我们选择鸢花数据（iris）进行方法的尝试训练，该数据集一共包含5个变量，其中4个特征变量，1个目标分类变量。共有150个样本，目标变量为花的类别分别是山鸢尾 (Iris-setosa)，变色鸢尾(Iris-versicolor)和维吉尼亚鸢尾(Iris-virginica)。包含的三种鸢尾花的四个特征，分别是花萼长度(cm)、花萼宽度(cm)、花瓣长度(cm)、花瓣宽度(cm)，这些形态特征在过去被用来识别物种。

对于目标变量，为了方便表示，通常用 0，1，2 .... 来代替不同的类别。所以在本数据集中，0，1，2分别代表'setosa', 'versicolor', 'virginica'三种不同花的类别。

在dataframe中，可以使用apply( )方法结合匿名函数的用法来修改某列的值：

***iris_all['target'].apply(lambda x : 'Iris-setosa' if x == 0 else ('Iris-versicolor' if x == 1 else 'Iris-virginica'))***

为了更直观的显示特征之间的关系，seaborn库的 pairplot方法提供了可视化探索数据特征之间的关系的方法：

***sns.pairplot(data=iris_all,kind="scatter",diag_kind='hist', hue= 'target')***  
***plt.show()***

（diag_kind用于控制对角线上图的类型，kind用于控制非对角线上图的类型，hue针对某一字段进行分类）


![pairplot](pairplot.png)

其中，对角线上是各个属性的直方图（分布图），而非对角线上是两个不同属性之间的相关图。从图中可发现，花瓣的长度和宽度之间以及萼片的长短和花瓣的长、宽之间具有比较明显的相关关系。



