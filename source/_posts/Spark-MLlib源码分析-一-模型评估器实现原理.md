---
title: 'Spark MLlib源码分析(一): 模型评估器实现原理'
date: 2022-06-07 08:49:33
tags: [Spark,Spark MLlib]
description: 图文拆解模型评估器实现原理
---

在[Spark MLlib分类模型评估最佳实践](https://tech.xiaomi.com/#/pc/article-detail?id=15440)一文里，我们实践了分类模型评估器BinaryClassificationMetrics的使用。但它是如何做到这么高效的构建混淆矩阵以及评估不同指标的，分箱是怎么回事？ 等等，我们还不太清楚。  本文就从源码角度分析其实现细节

# 为什么要了解评估器

​         我们知道分类模型的准召率是衡量模型好坏很重要的指标，这两个指标在某种程度上是彼此矛盾的，一般情况高查准率意味着低召回率，反之亦然。不同的应用场景，我们需要对准召率做一个权衡，对于逻辑回归这类输出概率的分类模型来说，调整阈值即可平衡准召率。每给定一个阈值，就去算一个准召率未免太麻烦，然而利用SparkMLlib内置的模型评估器就可以快速批量计算不同阈值下的准召率。 就像我们在[Spark MLlib分类模型评估最佳实践](https://tech.xiaomi.com/#/pc/article-detail?id=15440)一文里说的，如果不了解是怎么实现的，那么在使用过程中它输出的结果可能会令人困惑。这篇文章会详细讲解它的实现思想以及源码，并用图解的方式直观展现。

# 设计评估器？

​         要设计评估器，我们首先要理清楚评估器主要负责什么，它的输入、输出分别是什么？与模型之间有什么关系。 粗略的看，模型、评估器、指标之间是这么个关系：

![流程图](https://piggo-picture.oss-cn-hangzhou.aliyuncs.com/%E6%B5%81%E7%A8%8B%E5%9B%BE.jpg)

但其实评估器不关心模型具体是什么模型，模型只要把样本阳性"概率"给出来，再定一个阈值就完事儿了。评估器拿阈值和"概率"算出预测标签和实际标签对比构建**混淆矩阵**，然后你要什么指标，就传给评估器对应指标的公式，然后评估器就可以算出指标了。但是作为一个成熟的评估器，这些公式不应该用户传，最好是内置，只暴露API就行。

> 这里概率加引号，是因为有些场景，模型输出的概率并非通常意义下的概率，它不代表可能性。

这么一想，它们之间的关系应该是下面这样：

![流程图 (1)](https://piggo-picture.oss-cn-hangzhou.aliyuncs.com/%E6%B5%81%E7%A8%8B%E5%9B%BE%20(1).jpg)

综上： 

1. 评估器的输入是模型预测的概率和对应真实的标签
2. 评估器内置各种指标的计算公式
3. 评估器选择不同阈值，构造不同的混淆矩阵，根据公式计算相应指标

这里的难点是第三点，阈值怎么选？如何快速构建混淆矩阵？我们看Spark是如何实现的

# 源码解读

##  整体框架

首先，我们看这些指标的API，大多没有参数，且内部都调用了createCurve接口，只是参数不同：

```scala
  val metrics = new BinaryClassificationMetrics(predictionAndLabel)

  def fMeasureByThreshold(): RDD[(Double, Double)] = fMeasureByThreshold(1.0)

@Since("1.0.0")
  def fMeasureByThreshold(beta: Double): RDD[(Double, Double)] = createCurve(FMeasure(beta))
  /**
   * Returns the (threshold, precision) curve.
   */
  @Since("1.0.0")
  def precisionByThreshold(): RDD[(Double, Double)] = createCurve(Precision)

  /**
   * Returns the (threshold, recall) curve.
   */
  @Since("1.0.0")
  def recallByThreshold(): RDD[(Double, Double)] = createCurve(Recall)
```

我们还发现createCurve的返回结果就是指标，没有多余的逻辑，createCurve源码如下：

```scala
  private def createCurve(y: BinaryClassificationMetricComputer): RDD[(Double, Double)] = {

    confusions.map { case (s, c) =>//这里的confusions一会儿我们会发现它就是混淆矩阵

      (s, y(c))

    }

  }
```

先看参数： 发现它接受一个BinaryClassificationMetricComputer类型的参数，实际上上面三个参数类型都是它的子类型:

![img](https://xiaomi.f.mioffice.cn/space/api/box/stream/download/asynccode/?code=M2JlZDU3NTYzNDc1ZjZkMzNiNWQ2ZjZmY2M4NzgzMzRfZlpQaHUzVWRIV2ZWdTIxRndINE9sOGszOVdzZmxQcmNfVG9rZW46Ym94azRHYUNsNWV6QUwycGJYdkdlUmQ5b3VoXzE2NTU2Mjk4Mjg6MTY1NTYzMzQyOF9WNA)

从它们的名字看，这不就是指标计算公式嘛！接下来我们以精确率的计算为例来解读。先看Precision参数：

```scala
private[evaluation] object Precision extends BinaryClassificationMetricComputer {
  override def apply(c: BinaryConfusionMatrix): Double = {
   // totalPositives = TP + FP
    val totalPositives = c.weightedTruePositives + c.weightedFalsePositives
    if (totalPositives == 0.0) {
      1.0
    } else {
      // Precision = TP / (TP + FP)
      c.weightedTruePositives / totalPositives
    }
  }
}
```

和我们想的一样，它正是精确率计算公式，而它的参数就是混淆矩阵。再联想到precisionByThreshold返回的RDD，第一列是threshold，第二列是precision；这样confusions的结构就清晰了：它的第一列是阈值threshold,第二列是该阈值对应的混淆矩阵。

## 增量构建混淆矩阵源码解读

我们直接看confusion的构建源码， 重点看红色注释

```scala
confusions: RDD[(Double, BinaryConfusionMatrix)]) = {
    // Create a bin for each distinct score value, count weighted positives and
    // negatives within each bin, and then sort by score values in descending order.
   //按照预测概率值分组聚合
   //(score, BinaryLabelCounter) 每个预测值(降序排列)，都统计其正负样本数,是根据label计算的
    val counts = scoreLabelsWeight.combineByKey(
      createCombiner = (labelAndWeight: (Double, Double)) =>
       //统计正样本数， 负样本数
        new BinaryLabelCounter(0.0, 0.0) += (labelAndWeight._1, labelAndWeight._2),
      mergeValue = (c: BinaryLabelCounter, labelAndWeight: (Double, Double)) =>
        c += (labelAndWeight._1, labelAndWeight._2),
      mergeCombiners = (c1: BinaryLabelCounter, c2: BinaryLabelCounter) => c1 += c2
    ).sortByKey(ascending = false) //保证了分区内和分区间都有序

   // binnedCounts的数量跟numBins有关
    val binnedCounts =
      // ==0不分箱
      if (numBins == 0) {
        // 如果numBins==0也就是不分箱，则binnedCounts就是所有不同的预测值，这也就是为什么我们不分箱时，产生的结果非常大的原因。强烈建议分箱！
        counts
      } else {//分箱
        val countsSize = counts.count()//预测值去重后的个数
        // Group the iterator into chunks of about countsSize / numBins points,
        // so that the resulting number of bins is about numBins
        //每个箱子的大小
        val grouping = countsSize / numBins
        if (grouping < 2) {
          // numBins was more than half of the size; no real point in down-sampling to bins
          logInfo(s"Curve is too small ($countsSize) for $numBins bins to be useful")
          counts
        } else {
          //注意到mapPartitions的调用，这里是每个分区都这么分箱，这也解释了，为什么不对参数RDD重分区时，结果集可能会超出分箱参数的原因
          counts.mapPartitions { iter =>
            if (iter.hasNext) {
              var score = Double.NaN
              var agg = new BinaryLabelCounter()
              var cnt = 0L
              iter.flatMap { pair =>
                score = pair._1
                agg += pair._2
                cnt += 1
                if (cnt == grouping) {
                  /** 箱子最后一个预测值是这个箱子的最小的预测值，超过这个预测值的都被累计了
                  ret的第一列时该分区内的阈值，agg是对应的混淆矩阵；
                  */
                  val ret = (score, agg)
                  //清空计数器，返回混淆矩阵
                  agg = new BinaryLabelCounter()
                  cnt = 0
                  Some(ret)
                } else None
              } ++ {
                if (cnt > 0) {
                  Iterator.single((score, agg))
                } else Iterator.empty
              }
            } else Iterator.empty
          }
        }
      }
  //计算每个分区的统计值
   val agg = binnedCounts.values.mapPartitions { iter =>
    val agg = new BinaryLabelCounter()
    //这里的实现其实就是正负样本累加
      iter.foreach(agg += _)
      Iterator(agg)
    }.collect()  //把每个分区的计数收集到一起
   
   //计算分区间累加的统计值
   //scanLeft产生了一个数组，聚合所有分区的计数，长度是分区数+1
    val partitionwiseCumulativeCounts =
      agg.scanLeft(new BinaryLabelCounter())((agg, c) => agg.clone() += c)
   
    val totalCount = partitionwiseCumulativeCounts.last
    logInfo(s"Total counts: $totalCount")
   
   //part内累积：每个score先整体累加前一个part，在累加part内其他score的
    val cumulativeCounts = binnedCounts.mapPartitionsWithIndex(
      (index: Int, iter: Iterator[(Double, BinaryLabelCounter)]) => {
        //先累加上一个分区的统计值， 再逐个累加本分区内所有箱子的统计值
        val cumCount = partitionwiseCumulativeCounts(index)
        iter.map { case (score, c) =>
          cumCount += c
          (score, cumCount.clone())
        }
      }, preservesPartitioning = true)
    cumulativeCounts.persist()
   //计算混淆矩阵， 有了混淆矩阵，那些指标根据我们传的公式就可以计算指标了
    val confusions = cumulativeCounts.map { case (score, cumCount) =>
      (score, BinaryConfusionMatrixImpl(cumCount, totalCount).asInstanceOf[BinaryConfusionMatrix])
    }
    (cumulativeCounts, confusions)
  }
```

## 图解混淆矩阵增量构建过程

分箱之前，先构造了每个预测值对应的正负样本统计：

![流程图 (2)](https://piggo-picture.oss-cn-hangzhou.aliyuncs.com/%E6%B5%81%E7%A8%8B%E5%9B%BE%20(2).jpg)

当数据量很大时，模型预测结果(score)通常也很大，如果不分箱，计算结果将非常大！



分箱时，先计算每个箱子的大小，然后每个分区都进行分箱操作，每个箱子计算一个(预测值，样本统计)

预测值取最后一个预测值，由于事先是按降序排序的，所以这个预测值是当前最小的那个预测值，也就是说超过这个值的正负样本已经分好箱了。

![流程图 (3)](https://piggo-picture.oss-cn-hangzhou.aliyuncs.com/%E6%B5%81%E7%A8%8B%E5%9B%BE%20(3).jpg)

针对分箱后的结果，我们希望每个箱子都构建一个对应的全集的混淆矩阵，由于我们事先对数据进行了全局排序， 只需要逐个累加每个箱子：累加一个箱子后的混淆矩阵，其阈值也就是预测值是该箱子对应的预测值，正负样本统计值是大于该阈值的所有箱子的统计值累加。 这个累加的过程设计的比较巧妙：

先计算一个中间结果：

1. 每个分区内的箱子先分区内聚合， 对应代码中的agg
2. 再按照分区依次聚合,对应代码中的partitionwiseCumulativeCounts

当要计算累加某个箱子时，先从中间结果中查找该箱子所在分区的上一个分区的统计值，然后遍历当前分区的箱子并逐个累加，最后一个箱子的预测值就是当前混淆矩阵的阈值，累加结果就是该箱子对应的混淆矩阵。

图解如下：

![流程图 (4)](https://piggo-picture.oss-cn-hangzhou.aliyuncs.com/%E6%B5%81%E7%A8%8B%E5%9B%BE%20(4).jpg)

至此，完成了混淆矩阵组的构建

##  指标计算

其实完成了混淆矩阵组的构建，计算指标就非常简单，由上面提到的

BinaryClassificationMetricComputer负责计算，例如准确率:

```scala
private[evaluation] object Precision extends BinaryClassificationMetricComputer {
  override def apply(c: BinaryConfusionMatrix): Double = {
    val totalPositives = c.weightedTruePositives + c.weightedFalsePositives
    if (totalPositives == 0.0) {
      1.0
    } else {
      c.weightedTruePositives / totalPositives
    }
  }
}
```

# 总结

1. BinaryClassificationMetrics的实现最大的计算量是排序过程，分箱之后，每个混淆矩阵的计算都非常简单，不断复用前面的统计结果，这是整个过程高效的主要原因。
2. 不分箱会导致计算量和计算结果巨大，因此Spark对指标结果也设计为RDD。当然，除非数据量本身就比较小，可以不分箱，大多数情况下建议分箱
