# tf-faster-rcnn
# 本实验使用tensorflow作为深度学习框架
1.数据集
	1.1将图片文件和与图片对应的XML标注文件放置于 data\VOCDevkit2007\VOC2007\ 中 
	1.2利用 data\VOCDevkit2007 中的 test.py 生成 测试集、训练集、验证集 的txt文件

2.训练
	2.1通过以下命令开始训练：
		python train.py
	2.2通过修改 lib\config 中的 config.py 修改训练参数
	2.3训练好的网络权重文件存放在 output/ 中
	2.4本训练使用预训练的VGG16网络进行
3.测试
	3.1将测试数据集放到 data\demo\ 中
	3.2通过以下命令利用最新训练的权重对文件夹中的图片进行测试:
		python demo.py

