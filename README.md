# chinese_text_classification
通过一个中文文本分类问题系统实现了各种分类方法
# 数据来源
数据来源于搜狗新闻
# 类别
有***car，entertainment，military，sports，technology***五种类别。  
原始数据比较大，没有上传，分词，去除停用词之后的数据放在processed_data文件夹下。  
# 分类算法
主要实现了以下分类算法:    
- NB(朴素贝叶斯）
- SVM（支持向量机）
- fasttext
- text_CNN
- text_RNN
- text_RCNN
- text_Bi_LSTM
- text_Attention_Bi_LSTM
- HAN(Hierarchical Attention Network)
- ELMo
# 分类准确率
分类准确率都在90%附近，没有进行太多预处理，只为熟悉算法的使用。
# 依赖库
基于***tensorflow2.0***实现，可以在win和linux下运行。觉得有用的点个赞，谢谢。
