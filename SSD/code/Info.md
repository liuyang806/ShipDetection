# SSD: Single Shot MultiBox Detector in TensorFlow
***本实验使用tensorflow作为深度学习框架***
## 1.数据集
	1.1以VOC2007格式将训练图片集保存至 VOC2007/ 中
	1.2用以下命令生成训练集.tfrecord文件：
		python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=./VOC2007/ --output_name=voc_2007_test --output_dir=./tfrecords/

## 2.训练
	2.1通过以下命令开始训练：
		python train_ssd_network.py --train_dir=./train_models --dataset_dir=./tfrecords --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=./checkpoints/ssd_300_vgg.ckpt --checkpoint_exclude_scopes =ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --batch_size=8 --gpu_memory_fraction=0.6
		
		python train_ssd_network.py 
			--train_dir=./train_models
			--dataset_dir=./tfrecords 
			--dataset_name=pascalvoc_2007 
			--dataset_split_name=train 
			--model_name=ssd_300_vgg 
			--checkpoint_path=./checkpoints/ssd_300_vgg.ckpt 
			--checkpoint_exclude_scopes =ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box 
			--save_summaries_secs=60 
			--save_interval_secs=600 
			--weight_decay=0.0005 
			--optimizer=adam 
			--learning_rate=0.001 
			--batch_size=8#本机batch_size过大会显存不够
			--gpu_memory_fraction=0.6
			
			--snapshot=train_models/model.ckpt-xxxx  （继续未完成的训练）
	2.2通过修改train_ssd_network.py可修改：
		max_number_of_steps（第154行）
	2.3训练模型存放在 train_models/ 中

## 3.测试
	3.1将测试数据集放到 VOC2007/demo 中
	3.2运行 notebooks/ 中 ssd_test.py 对测试数据集进行测试

## 4.Tensorboard
	在项目主目录运行 tensorboard --logdir=train_models