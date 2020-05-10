# YOLOv3: You Only Look Once v3 in darknet
***本实验使用darknet框架完成YOLOv3训练***
## 1.数据集
	1.1将图片文件和与图片对应的txt文件放置于 train/ 中 （txt文件由voc_label.py通过xml文件生成）
	1.2利用 C:\Users\liuyang\Desktop\Graduation Project\ShipDetection\code\YOLOv3 中的 trainAndTest.py 生成 train_file.txt 与 vaild_file.txt

## 2.训练
	2.1通过以下命令开始训练：
		build\darknet\x64\darknet.exe detector train .\cfg\ship.data .\cfg\yolov3-ship.cfg .\build\darknet\x64\darknet53.conv.74 >> yolov3.log
	2.2通过以下命令在上次训练的基础下继续训练模型：
		build\darknet\x64\darknet.exe detector train .\cfg\ship.data .\cfg\yolov3-ship.cfg backup\yolov3-ship_last.weights >> yolov3.log
	2.3通过修改 ./cfg 中的 yolov3-ship.cfg 进行网络模型配置
	2.4训练模型存放在 backup/ 中

## 3.测试
	3.1将测试数据集放到 testImg/ 中
	3.2通过以下命令利用最新训练的权重进行图片测试:
		build\darknet\x64\darknet.exe detector test .\cfg\ship.data .\cfg\yolov3-ship.cfg backup\yolov3-ship_last.weights

## 4.其他
	在项目主目录运行 yolov3.log 查看训练的 log 信息
	主目录中的chart.jpg为loss函数降低图像