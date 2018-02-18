# Kaggle竞赛《泰坦尼克生存预测》学习笔记

# 简介
https://www.kaggle.com/c/titanic

泰坦尼克生存预测是Kaggle专门设置的一个竞赛**入门练习**，它以竞赛形式体现，但不计成绩，竞赛开始至今已经两年，目前有9531支团队参赛，我两天前提交的成绩还在6%的位置，两天后就后退到8%，可见目前该竞赛还很活跃。竞赛目标是根据891条训练数据生成模型，预测400多条测试集数据的生存情况。

Kernels是大家交流发现的地方，基于此竞赛的学习特性，本Kernels中有超过40篇铜牌及以上教学性质的notebook，可谓争奇斗艳，本人趁着假期在kernels中选取5篇有各自特点的notebook进行学习并做此笔记以备之后查阅。

# Notebook
选取的五篇notebook如下：

* [Titanic Data Science Solutions
](https://www.kaggle.com/startupsci/titanic-data-science-solutions) 数据的基础处理，可视化，解释详尽，提交得分0.77左右，893个vote up(点赞)。
* [Introduction to Ensembling/Stacking in Python
](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python) 集成学习，提交得分0.80861，1278 vote up。
* [Roy's Titanic Notebook
](https://www.kaggle.com/gaohong/roy-s-titanic-notebook)高级特征提取，处理。提交得分0.813左右
* [Titanic 0.82 - 0.83](https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83) 第一句话是`This kernel is not for beginners`提交的分0.83
* [A Data Science Framework: To Achieve 99% Accuracy
](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook) 炫技派？整个体系最为全面，完整，涵盖参数选择，ensemble，CV，模型选择。宣称最高得分为0.88，但实际结果集提交后的分仅为0.77 （cheating? 背后原因值得深思）

Kaggle的notebook本来可以fork后直接云端运行，但是没有本机的jupyter notebook方便，其中前三篇都是在本机的jupyter notebook中进行的调试。[github地址](https://github.com/dashjim/kaggle-titanic)

notebook示例：

* [Titanic Data Science Solutions
](https://github.com/dashjim/kaggle-titanic/blob/master/Titanic/self_play.ipynb)


# Notebook 1. Titanic Data Science Solutions


## 亮点

* 大量使用seaborn来plot数据，比matplotlib简单。FacetGride简单强大，可惜survived/none survived分列在两个图中，不像后面将两个图叠加在一起。
* 在5个notebook中对数据的解释最为细致清楚，特征处理的代码极为简洁
* 使用分段方式来处理Age，这一点是本人之前不知道的。在后面的所有notebook中都大量使用各种方式来将数据处理成categorical。本人测试时将categorical数据恢复成原始数据后，模型训练得分大幅提升至0.93，但提交得分（测试得分）反而下降，说明数据分段有助于泛化。
* 从name字段中提取title属性，细致功夫让人叹服。
*  Age的缺失值填充方式，提出使用中位数(median)加groupby(pclass)而不是均值（mean）+标准差。

## 改进点

* 模型方面涉及到的知识点较少，使用一众模型都是比较经典的模型，没有使用ensemble，模型选择等。
* 训练时没有使用cross validation，模型的得分来直接自于`clf.score()`函数.
* 特征方面没有涉及到特征缩放（scaler），特征选择。没有提取高级特征如parent survival等。
* Age人为分段，没有使用`cut()`, `qcut()`

## 小结与思考
似乎这个notebook的作者故意为了保持简单性而省略掉大量的处理。得分不高，排在60%左右，但清晰易懂，特别适合初学者。搞清数据特征之后学习后面的notebook就轻松了。

Age缺失值得处理，所有5篇notebook中使用的都是类似的基于中位数的处理，但是对于数据的刻画指标按准确度排序应为 _均值<中位数<期望<分布_ 。是否可以使用期望，分布，甚至模型预测的方式来填充缺失值呢？

# Notebook 2. Introduction to Ensembling/Stacking in Python

## 亮点小结

* 上个notebook中尝试了多个不同的模型，确没有进一步ensemble，比较遗憾。其它notebook中多有使用简单的vote based ensemble，但效果并不好。这篇作为vote up得分冠军，个人觉得是因为使用基于stacking的ensemble，而且取得了较好的效果。
* 集成学习-ensemble，之前大致了解原理，但从没真正搞明白，现在才明白这是个大的概念，有些是独立的算法，如随机森林是集成了多个决策树，在sklearn中也是归到ensemble下的。有些是将不同类的算法预测结果再次进行提升，如voting，stacking。
* 有文章指出本文使用的stacking可能是所有ensemble方法中效果最好的。
* stacking具体过程为先用不同算法基于Kfold进行训练和CV。将CV时的预测值保存下来，每个算法的预测结果为一列（一个feature），多个算法的多组预测值（基于训练集）合在一起即是tier1的输出，用作tier2算法的输入，来训练一个新的模型（m2）。tier1中的算法还要预测（处理）测试集，所产生的多组预测结果通过`mean()`来缩小为一组结果(test2)。新的模型（m2）对test2进行预测的结果即为最终提交结果。
* seaborn的`Pearson Correlation Heatmap`用来展示特征间的相关系数，实用。
* `qcut()`处理Fare，`cut()`处理Age，这是基于字段的业务含义来选择的。
* **Feature权重**是模型训练后输出的。`rf_feature = rf.feature_importances(x_train,y_train)
`，使用`import plotly.graph_objs as go
`动态图来可视化。这些权重只能用来解释数据（feature），并不影响得分和模型训练。
* 首次接触xgboost这一Kaggle重器，细节有待学习。

## 思考
* 使用xgboost作为tier2模型，没有进行参数选择，如果使用参数选择的话分数会更高？

# Notebook3. Roy's Titanic Notebook
## 亮点
* 5个notebook里面特征工程做的最好的，提取出了`male/female Friends survived`，`MotherSurvived `，`ChildOnBoard `，`ChildSurvived `，`FatherOnBoard`，`MotherOnBoard `，等一众feature。
* 使用最基本的matplotlib，子视图可视化使用不花哨。
* 模型参数选择+特征选择 - `from sklearn.feature_selection import SelectKBest, f_classif`, `from sklearn.model_selection import GridSearchCV`
* 首次使用基于keras框架的神经网络，预测得分在本notebook所使用的模型中最高（beat out RF and XGB）。需要进一步学习这个框架。
* 有使用遗传算法`genetic program`，得分还不错。这个是个难得的我比较清楚的小众算法。遗传算法简称GP，缩写与高斯过程（Gaussian Process）重名，后者完全不懂（只看过Andrew NG视频中的高斯kernel），然后当时一直误以为notebook中使用的是高斯过程，还去看了半个小时的视频教程。
* 最好的特征工程+厉害的算法=0.813

# Nobebook4. Titanic [0.82] - [0.83]
* 又一个高分。没废话，一上来综述特征上的发现，直接给出结论。
* 介绍Kaggle的public score和private score，指出有些情况下会出现过拟合public score的情况。

> if you submit your results too many times, you subconsciously "bleed" public test set data into your models, and your models adapt to the public test set a little more. They may tend to overfit to the public test set, meaning your score on a private test set may drop significantly. 

* 指出处理Age缺失值时中位数优于均值是因为中位数受极值和分布异常的影响较小。
* 和前面notebook不同，做分段处理时Age，Fare都是使用基于频率的`qcut()`。
* 唯一指出特征缩放的重要性，使用特征缩放`Standard Scaler`

> Standard Scaler is our friend. It helps to boost the score. Scaling features is helpful for many ML algorithms like KNN for example, it really boosts their score. 

* KNN这一原理上最为简单的模型取得最高分8.3+
* 作者认为这一数据集真正的高分是0.86+

> The truly better model would yield 86-88+, but it's very tough to get there.

# Notebook5. A Data Science Framework: To Achieve 99% Accuracy

## 重点

* 过程大而全，能有的都有了。
* 自己手写了一个经验模型，效果也不错。
* IN[29]决策树可视化
* `ensemble.VotingClassifier `基于voting的集成学习。
* 暴力计算 - 在循环中先对每一个算法进行参数选择，最后再利用选择出的参数进行voting的集成学习。
* `LabelEncoder()`用来生成特征`'Embarked_Code'`
* gridSearch使用roc_auc作为模型评判标准`model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)`
* `seaborn.FacetGrid()`可视化
* In[26]`confusion matrix` 混淆矩阵
* In[28] 特征选择recursive feature elimination (RFE) with cross validation (CV) `feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)`
* 作者是标题党，并有作弊嫌疑，notebook显示最高分为0.88，但本人提交里面预测集的得分为0.77左右，作者没有指出他说的得分到底是交叉验证的得分还是提交得分（public score）。本人留言中提出质疑，尚无回复。
* 好模型不如好数据。留言中有关于为啥测试集public score这么低的讨论，似乎指向过度训练。（后记，[Stanford一个关于boosting的PPT](http://jessica2.msri.org/attachments/10778/10778-boost.pdf)-28页中指出训练数据中存在高斯噪声时会出现越训练测试集得分越低，训练集得分越高的情况）

# 总结

* **从数据看绅士精神** - 下方数据显示女性的生存概率远大于男性（年龄最大的人都生存下来了），电影中的“让女性和老人儿童先上船”，不是虚构的。

1. **Title	Survived**
2. Master	0.575000
3.	Miss	0.702703
4.	Mr	    0.156673
5.	Mrs 	0.793651
6.	Other	0.347826

* **从数据看权贵的生存几率**，商务舱（pclass=1）的生存概率三倍于经济舱（pclass=3），关键时刻喊的应该是"让_商务舱客人，_女性和老人儿童先上船"！- 纯属玩笑，据说真正的原因是商务舱配备的救生艇比较多。

0.	**Pclass	 Survived**
1.	1	0.629630
2.	2	0.472826
3.	3	0.242363

* 尽管绝大多数参赛者使用了交叉验证（Cross Validation）来防止过拟合，但所有notebook都存在训练集得分大幅超越提交实际得分的情况。说明CV也有其局限性。
* 是否可以使用深度神经网络自行生成大量组合特征（数百量级）并提取有效特征？
* 高分只是竞赛的结果，对高分的追求过程中的学习，思考，假设，试验是更有意义的地方。
* No free lunch理论指出，没有最牛的模型，只有最合适的模型。
* 后续阅读-kernel中提到的 -[Stanford一个关于boosting的PPT](http://jessica2.msri.org/attachments/10778/10778-boost.pdf)-
* 后续学习其它典型类别的正式比赛如推荐，无监督，股市预测等。
* 整理模型评估指标的异同：accuracy, percision, recall, confusion matrix, roc, auc。
* 学习各种pipeline用法。
