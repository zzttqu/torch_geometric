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
* pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
# 模型问题
* 图神经网络的输出是每个节点的信息，所以最后的线性层也改成每个节点就好了，不用细分了
* 输出层能不能跟随要求改变节点数量就好了
* 目前只能跑目标100产品左右的，需要在较短步数内完成，要不然会崩掉摆烂
* 准备搞异质图
## 生成requirements.txt
* 需要忽略虚拟空间
* pipreqs ./ --encoding UTF-8 --ignore ".venv"