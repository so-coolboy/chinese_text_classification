## chinese_text_classification
通过一个中文文本分类问题系统实现了各种分类方法


数据来源于搜狗新闻，有***car，entertainment，military，sports，technology***五种类别，  
原始数据比较大，没有上传，分词，去除停用词之后的数据放在processed_data文件夹下。  
主要实现了以下分类算法:  
***NB， SVM， fasttext,  text_cnn,  text_rnn,  text_rcnn,   
text_bi_lstm,  text_attention_bi_lstm,  HAN(Hierarchical Attention Network),  ELMo***.    
分类准确率都在90%附近，没有进行太多预处理。
基于***tensorflow2.0***实现，可以在win和linux下运行。觉得有用的点个赞，谢谢。
