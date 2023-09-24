每个工作单元有四个状态，

* 0号位置是异常状态，
* 1号位置是当前功能，
* 2号位置是当前功能的速度，
* 3号位置是当前功能的原料数量
* 4号位置是当前功能的产品数量

功能有

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