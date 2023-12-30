每个工作单元有四个状态，

* 0号位置是异常状态，
* 1号位置是当前功能，
* 2号位置是当前功能的速度，
* 3号位置是当前功能的原料数量
* 4号位置是当前功能的产品数量

功能有

# 现在纠结到底怎么做动作空间，是工作中心动作，还是工作单元动作

感觉现在就是这样，工作中心一个动作，动作空间为工作单元数量+1，表示启动哪个工作单元，工作单元一个动作，是否接收原材料

# ——————

* 0是停止当前工作
* 1是运行哪个功能，细节是运行功能的代号
* 0是停止从区域获取原材料
* 1是开始获取，各个获取权重一样
* 还有从暂存中心获取原材料的权重

基础设定

* 每个加工单元之多存放一个step的生产产品
* 所有加工产品在加工step完成后都会运输到集散点
* 集散点可以存储产品并按权重发送到下一个程序
* 安装图神经网络优化包
* pip install torch_scatter torch_sparse torch_cluster torch_spline_conv
  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# 模型问题

* 图神经网络的输出是每个节点的信息，所以最后的线性层也改成每个节点就好了，不用细分了
* 输出层能不能跟随要求改变节点数量就好了
* 目前只能跑目标100产品左右的，需要在较短步数内完成，要不然会崩掉摆烂
* 准备搞异质图

## 生成requirements.txt

* 需要忽略虚拟空间
* 安装服务器
* pip install -U channels["daphne"]
* pipreqs ./ --encoding UTF-8 --ignore ".venv"
* pipreqs --ignore .venv --force --encoding=utf-8

## 我的设定

### 原则

1. 一款产品有多道工序
2. 每个产品的工序可以由同一单元执行，但是执行内容不同
   > 例如：两种减速器，外壳不同，但是外壳都需要铸造。减速比不同，但是都需要装配齿轮。
   > 这个时候就可以使用同一个铸造单元和同一个装配单元执行不同的工序，也是如何调配工作单元，排产的问题
3. 一个可重构单元可以执行多道工序


* 程序干两件事
* 第一件事，用遗传算法，根据工作单元数量和种类，给每个产品，分配一个基础生产线。
* 第二件事，在执行订单时对加工单元进行适当重构

* 订单列表
  > 优化目标是整个订单用时最短
  >> 可分配单元数：A设备16，B设备10，C设备10，D设备15，E设备10

| 订单编号 | 产品0数量 | 产品1数量 | 产品2数量 |   |
|:----:|:-----:|:-----:|:-----:|---|
|  0   |  10   |  20   |   4   |   |
|  1   |  60   |   5   |   5   |   |
|  2   |  20   |  10   |  15   |   |

* 产品流程
  > A是连接线缆，B是检查线缆连接情况，C是装配其他零件，D是装配外壳，E是检查整体状态

| 产品 | 工序1 | 工序2 | 工序3 | 工序4 | 工序5 | 用时 |
|----|-----|-----|-----|-----|-----|----|
| 0  | A   | B   | C   | D   | E   | 62 |
| 1  | A   | B   | C   | E   | -   | 42 |
| 2  | A   | B   | D   | E   | -   | 47 |

* 工序耗时
  > 切换不同产品的工序时用时为1个时间步

| 单元种类    | A | B  | C  | D  | E  |
|---------|---|----|----|----|----|
| 1产品所用时间 | 5 | 10 | 15 | 20 | 12 |
| 2产品所用时间 | 8 | 12 | 18 | —  | 12 |
| 3产品所用时间 | 3 | 6  | -  | 10 | 8  |

