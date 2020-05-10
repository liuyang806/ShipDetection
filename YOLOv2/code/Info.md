# YOLOv2: You Only Look Once v2 in darkflow
***本实验使用darkflow将darknet转化为tensorflow完成YOLOv2训练***
## 1.数据集
	1.1将图片文件和与图片对应的XML标注文件放置于 train/ 中 

## 2.训练
	2.1通过以下命令开始训练，本次训练使用YOLOv2的预训练权重进行：
		python ./flow --model cfg/yolo-3c.cfg --load bin/yolo.weights --train --annotation train/Annotations --dataset train/Images --gpu 1.0
	2.2通过以下命令在上次训练的基础下继续训练模型：
		python ./flow --model cfg/yolo-3c.cfg --load -1 --train --annotation train/Annotations --dataset train/Images --gpu 1.0
	2.3通过修改 ./cfg 中的 yolo-3c.cfg 进行网络模型配置
	2.4训练模型存放在 ckpt/ 中

## 3.测试
	3.1将测试数据集放到 sample_img/ 中
	3.2通过以下命令利用最新训练的权重进行图片测试:
		python ./flow --imgdir sample_img/ --model cfg/yolo-3c.cfg --load -1 --gpu 1.0
