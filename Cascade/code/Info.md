# LBP特征的cascade分类器
***本实验使用opencv自带分类器程序完成(opencv_createsamples.exe、opencv_traincascade.exe)***
## 1.数据集
	1.1正负样本的准备
		正样本：
			统一尺寸的舰船目标图像集，存放于pos文件夹中，并生成正样本目录文件，数据格式如下：
				pos/pos_001.jpg 1 0 0 45 45
		负样本：
			负样本为不含正样本的背景图像，且数据量约为正样本的三倍，生成负样本目录，结构如下：
				neg/neg_001.jpg
	1.2利用opencv_createsamples.exe生成正样本描述文件pos.vec，命令如下：
		opencv_createsamples.exe  -vec pos.vec  -info pos.txt -num 371 -w 45 -h 45 pause
## 2.训练
	2.1通过以下命令开始训练：
		opencv_traincascade.exe -data xml0.5 -vec pos.vec -bg neg.txt -numPos 350 -numNeg 800 -numStages 20 -precalcValBufSize 2048 -precalcIdxBufSize 1024 -w 45 -h 45 -minHitRate 0.995 -maxFalseAlarmRate 0.5 -weightTrimRate 0.95 -featureType LBP
		pause
	2.3训练好的XML分类器文件存放在 XML/ 中
	2.4本次训练使用LBP特征
## 3.测试
	3.1将测试数据集放到 testResult\test\ 中
	3.2在 testResult\ 中通过以下命令利用 cascade.xml 对文件夹中的图片进行测试:
		python test.py

