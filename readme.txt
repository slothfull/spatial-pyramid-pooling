@1 divide_spatial_pyramid_pooling.py 
-> 构建自己的SPPnet结构函数 
   可以将给定的feature_map 划分成任意mxn的sub_featuremap并且针对每一个subfmap进行pooling操作 获得mxn的pooling结果
   
@2 divide_spatial_pyramid_pooling_bp_test.py 
-> 测试自己实现的类似SPPnet的结构的性能

@3 我所实现的spatial pyramid pooling的操作和caffe中以及tensorflow中的方式不同，它们都是需要将一个feature map划分成相同尺度的大小，而我的feature map中划分是动态拼接的，不需要线性插值或者padding等操作，能够实现feature map利用率100%。
