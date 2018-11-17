# Task tracking

记录每天的工作，包括对项目进行了什么修改，发现了什么bug，哪些地方需要进行优化，甚至是期间想到的好的点子。**具体到每天写了什么代码，如：修改了某个函数，修改了具体那里，能起到什么作用；明天要对那个文件的什么部位进行修改，达到什么效果。**

## 18.11.9

- 计算卡方核的核矩阵具有负值，无法计算
- SVM对属性预测时进行的Platt scaling究竟时用来干什么的
- 测试时，一张图片进来进行各个属性的预测后，得到的概率怎么用于bayes的计算
- sklearn中的bayes具体怎么用，是用属性向量还是对应的属性概率向量作为训练时的输入

## 18.11.14

- 调整项目目录结构，增加tests文件夹和data文件夹
- 解决模块多层级引用 sys.path.appen('..')
- 将svm更改为LinearSVC，并进行输出准确性验证
- 采用最大后验概率预测图像类别
- 写出测试用例

***

- 跑测试用例，先调通
- attr_process.py中的属性读取和属性向量扩展功能应该分开 *done*
- utils.py中的get_class_name()方法返回数据不应该用list *wrong idea*
- evalute的调用太繁杂，晦涩

## 18.11.15

- 将attr_process.py中的属性读取和扩展功能分开
- 完整的跑一便修改了的算法
- 对main.py predict.py添加必要的提示性输出
- ***修改get_class_name()，更改返回值的形式*** ***wrong idea*** get_class_name()是要取得某个数据划分里类别的名字，直接用list保存很合适

***

- cls_predict()函数的参数不优雅
- 整个项目的代码有很多冗余，比如utils模块里的expand_attr()和attr_process()里的expand_attr()
- predict.py中的attr_predict()方法中有冗余，两次相似的功能，只是结果有点差异，应该合并
- 很多函数都需要传入dataset_path这个参数，怎样才能提取出来进行精简

## 18.11.16

- utils.py中的class_name_of_split()函数在读取类别名的时候需要注意最后一次加入列表的是一个空字符串，要进行检测
- attr_process.py中47行的all_class_name的长度是51，应该是50才对，而且57行取所有对应属性时下标超出了范围，原因就是上面的class_name_of_split()函数没修改之前读到了一个空的字符串

***

- 跑通代码

## 18.11.17

- 跑完代码，结果：AUC 83 准确率 58.9

***