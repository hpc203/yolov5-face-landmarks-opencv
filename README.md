# yolov5-face-landmarks-opencv
yolov5检测人脸和关键点，只依赖opencv库就可以运行，程序包含C++和Python两个版本的。
本套程序根据https://github.com/deepcam-cn/yolov5-face 里提供的训练模型.pt文件。转换成onnx文件，
然后使用opencv读取onnx文件做前向推理，onnx文件从百度云盘下载，下载
链接：https://pan.baidu.com/s/14qvEOB90CcVJwVC5jNcu3A 
提取码：duwc 

下载完成后，onnx文件存放目录里，C++版本的主程序是main_yolo.cpp，Python版本的主程序是main.py
。此外，还有一个main_export_onnx.py文件，它是读取pytorch训练模型.pt文件生成onnx文件的。
如果你想重新生成onnx文件，不能直接在该目录下运行的，你需要把文件拷贝到https://github.com/deepcam-cn/yolov5-face
的主目录里运行，就可以生成onnx文件。如果运行过程中没有报错中断，那就说明转换生成onnx文件成功，
并且opencv读取onnx文件做forward也正常。

我之所以发布这个文件，是因为在此之前我发布的一套使用opencv部署yolov5的程序里，转换生成onnx需要经过读取.pt文件生成.pth
文件和编写yolov5.py文件这两个步骤，这个挺麻烦的。然而最近我发现其实可以在读取.pt文件后直接生成.onnx文件的，这样就简化
了生成onnx文件的流程。

此外，我看到github上deepcam-cn更新了yolov5-face的程序，我的使用opencv做yolov5-face推理的程序也做了更新，
更新的程序发布在
https://github.com/hpc203/yolov5-face-landmarks-opencv-v2
