## sjtu_cv_course_homework
#### 介绍
上海交通大学2023年秋季课程--基于深度学习的计算机视觉课程大作业
#### 作业要求
##### hw1
>+ 数据集下载[链接](https://jbox.sjtu.edu.cn/l/Y168xQ)，请同学们自行实现dataloader以读取数据
>+ 请使用train文件夹中的数据训练模型，val文件夹中的数据测试模型，训练和测试时请对读取的数据进行shuffle
>+ 请计算训练好的模型在val set上的top-1与top-5 accuracy
>+ 可以fine-tune预训练好的Resnet模型，也可以自己train from scratch
>+ 请在模型训练时自行选择一些data augmentation方式使用，并比较data augmentation对accuracy的影响
>+ 请对训练过程中的loss curve和accuracy curve进行可视化，并完成报告
##### hw2
>+  数据集下载链接：[链接](https://jbox.sjtu.edu.cn/l/D1vaWa)，请同学们自行实现dataloader以读取数据
>+  请使名字中带有train的文件夹中的数据训练模型，val文件夹中的数据测试模型，数据集的注释存放在对应的文件夹里
>+ 请使用预训练好的Resnet模型作为backbone，自行实现detection head，并在数据集上完成object detection任务的训练与测试
>+ 请计算训练好的模型在val set上的mAP
>+ 请对训练过程中的loss curve和mAP curve进行可视化，并提供一些val set的可视化示例，完成报告