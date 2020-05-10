# ShipDetection
## 任务信息
	本项目通过级联分类器、 Faster R-CNN 、YOLOv2、YOLOv3和 SSD 算法对SAR图像进行舰船目标检测，  
	利用深度学习框架实现算法的调用，并调用 GPU 提升模型训练速度。通过训练集图像对算法框架进行  
	训练，并对训练的模型进行检测测试。  
## 数据集
	本实验目标检测项目集使用的是SAR雷达舰船图像，合成孔径雷达（SAR），是一种主动式的对地观测系  
	统，它能实现全天时、全天候对地实施观测、并具有一定的地表穿透能力。  
	本实验所用的原始SAR图像数据集共包含4万张jpg格式的图像及与每张图相对应的xml标注文件，标注文件中  
	包含了图像中所有舰船目标的位置信息。
	* 级联分类器的训练中，训练正样本为统一尺寸的舰船目标图像集，负样本为不含正样本的背景图像，图像数据均由原始的SAR图像通过截取获得。
	* Faster R-CNN 、YOLOv2模型的训练均使用VOC格式的训练集数据，通过原始SAR图像数据直接进行训练
	* YOLOv3利用已有数据集基础上生成YOLO的txt标注文件来标注每张图像中舰船的为位置信息
	* SSD通过原始数据集生成.tfrecord文件，用于模型训练
	![image](https://github.com/liuyang806/ShipDetection/SARimg.png)
	![image](https://github.com/liuyang806/ShipDetection/SARxml.png)
## 运行环境
	1.硬件环境

	名称|配置
	----|----
	处理器|Intel Core i7-6700HQ CPU@2.60GHz
	显卡|NVIDIA GeForce GTX 960M
	物理内存|8.00 GB
	操作系统|Windows 10 家庭中文版，64位
	编程语言|Python 3.5.6
	
	2.框架

	算法|学习框架
	----|----
	Faster R-CNN|tensorflow-gpu 1.13.1
	YOLOv2|darkflow
	YOLOv3|darknet
	SSD|tensorflow-gpu 1.13.1
	
	3.其他的相关软件和工具库主要有 CUDA10.0、NVIDIA cuDNN 7.5、Anaconda4.6、OpenCV3.4等

## 配置信息
	
***注；各模型具体训练流程见各算法 code/ 文件夹中的 Info.md***
***    本实验所用到的SAR图像数据、预训练模型及训练成熟的模型文件不包含在本项目中，具体地址为：***
***    ***