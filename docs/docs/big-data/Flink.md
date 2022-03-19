# Flink

## 1. Flink 简介

Flink 是一个分布式的流处理框架，它能够对有界和无界的数据流进行高效的处理。Flink 的核心是流处理，当然它也能支持批处理，Flink 将批处理看成是流处理的一种特殊情况，即数据流是有明确界限的。这和 Spark Streaming 的思想是完全相反的.

Spark Streaming 的核心是批处理，它将流处理看成是批处理的一种特殊情况， 即把数据流进行极小粒度的拆分，拆分为多个微批处理。

Flink 有界数据流和无界数据流：

![](https://image.ldbmcs.com/2020-10-25-jy6Wr5.jpg)

Spark Streaming 数据流的拆分：

![](https://image.ldbmcs.com/2020-10-25-7IE5ru.jpg)

## 2. Flink 核心架构

Flink 采用分层的架构设计，从而保证各层在功能和职责上的清晰。如下图所示，由上而下分别是 API & Libraries 层、Runtime 核心层以及物理部署层：

![](https://image.ldbmcs.com/2020-10-25-UAmaYT.jpg)

### 2.1 API & Libraries 层

这一层主要提供了编程 API 和 顶层类库：

* 编程 API : 用于进行流处理的 DataStream API 和用于进行批处理的 DataSet API；
* 顶层类库：包括用于复杂事件处理的 CEP 库；用于结构化数据查询的 SQL & Table 库，以及基于批处理的机器学习库 FlinkML 和 图形处理库 Gelly。

### 2.2 Runtime 核心层

这一层是 Flink 分布式计算框架的核心实现层，包括作业转换，任务调度，资源分配，任务执行等功能，基于这一层的实现，可以在流式引擎下同时运行流处理程序和批处理程序。

### 2.3 物理部署层

Flink 的物理部署层，用于支持在不同平台上部署运行 Flink 应用。

## 3. Flink 分层 API

在上面介绍的 API & Libraries 这一层，Flink 又进行了更为具体的划分。具体如下：

![](https://image.ldbmcs.com/2020-10-25-A65dz4.jpg)

按照如上的层次结构，API 的一致性由下至上依次递增，接口的表现能力由下至上依次递减，各层的核心功能如下：

### 3.1 SQL & Table API

SQL & Table API 同时适用于批处理和流处理，这意味着你可以对有界数据流和无界数据流以相同的语义进行查询，并产生相同的结果。除了基本查询外， 它还支持自定义的标量函数，聚合函数以及表值函数，可以满足多样化的查询需求。

### 3.2 DataStream & DataSet API

DataStream & DataSet API 是 Flink 数据处理的核心 API，支持使用 Java 语言或 Scala 语言进行调用，提供了数据读取，数据转换和数据输出等一系列常用操作的封装。

### 3.3 Stateful Stream Processing

Stateful Stream Processing 是最低级别的抽象，它通过 Process Function 函数内嵌到 DataStream API 中。 Process Function 是 Flink 提供的最底层 API，具有最大的灵活性，允许开发者对于时间和状态进行细粒度的控制。


## 4. Flink 集群架构


## 4.1  核心组件

按照上面的介绍，Flink 核心架构的第二层是 Runtime 层， 该层采用标准的 Master - Slave 结构， 其中，Master 部分又包含了三个核心组件：Dispatcher、ResourceManager 和 JobManager，而 Slave 则主要是 TaskManager 进程。它们的功能分别如下：

* **JobManagers** (也称为  *masters* ) ：JobManagers 接收由 Dispatcher 传递过来的执行程序，该执行程序包含了作业图 (JobGraph)，逻辑数据流图 (logical dataflow graph) 及其所有的 classes 文件以及第三方类库 (libraries) 等等 。紧接着 JobManagers 会将 JobGraph 转换为执行图 (ExecutionGraph)，然后向 ResourceManager 申请资源来执行该任务，一旦申请到资源，就将执行图分发给对应的 TaskManagers 。因此每个作业 (Job) 至少有一个 JobManager；高可用部署下可以有多个 JobManagers，其中一个作为  *leader* ，其余的则处于 *standby* 状态。
* **TaskManagers** (也称为  *workers* ) : TaskManagers 负责实际的子任务 (subtasks) 的执行，每个 TaskManagers 都拥有一定数量的 slots。Slot 是一组固定大小的资源的合集 (如计算能力，存储空间)。TaskManagers 启动后，会将其所拥有的 slots 注册到 ResourceManager 上，由 ResourceManager 进行统一管理。
* **Dispatcher** ：负责接收客户端提交的执行程序，并传递给 JobManager 。除此之外，它还提供了一个 WEB UI 界面，用于监控作业的执行情况。
* **ResourceManager** ：负责管理 slots 并协调集群资源。ResourceManager 接收来自 JobManager 的资源请求，并将存在空闲 slots 的 TaskManagers 分配给 JobManager 执行任务。Flink 基于不同的部署平台，如 YARN , Mesos，K8s 等提供了不同的资源管理器，当 TaskManagers 没有足够的 slots 来执行任务时，它会向第三方平台发起会话来请求额外的资源。

## 5. Flink 的优点

最后基于上面的介绍，来总结一下 Flink 的优点：

* Flink 是基于事件驱动 (Event-driven) 的应用，能够同时支持流处理和批处理；
* 基于内存的计算，能够保证高吞吐和低延迟，具有优越的性能表现；
* 支持精确一次 (Exactly-once) 语意，能够完美地保证一致性和正确性；
* 分层 API ，能够满足各个层次的开发需求；
* 支持高可用配置，支持保存点机制，能够提供安全性和稳定性上的保证；
* 多样化的部署方式，支持本地，远端，云端等多种部署方案；
* 具有横向扩展架构，能够按照用户的需求进行动态扩容；
* 活跃度极高的社区和完善的生态圈的支持。


## 参考资料

[](https://gitee.com/jerry-chen417/flink-real-time-data-warehouse/blob/master/Flink%E5%9F%BA%E7%A1%80.md)
