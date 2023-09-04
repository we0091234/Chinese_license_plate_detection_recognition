## What's New

**2022.12.04 车辆和车牌一起检测看这里[车辆系统](https://github.com/we0091234/Car_recognition)**

[yolov7 车牌检测+识别](https://github.com/we0091234/yolov7_plate)

[安卓NCNN](https://github.com/Ayers-github/Chinese-License-Plate-Recognition)

## **最全车牌识别算法，支持12种中文车牌类型**

**环境要求: python >=3.6  pytorch >=1.7**

#### **图片测试demo:**

直接运行detect_plate.py 或者运行如下命令行：

```
python detect_plate.py --detect_model weights/plate_detect.pt  --rec_model weights/plate_rec_color.pth --image_path imgs --output result
```

测试文件夹imgs，结果保存再 result 文件夹中

#### 视频测试demo  [2.MP4](https://pan.baidu.com/s/1O1sT8hCEwJZmVScDwBHgOg)  提取码：41aq

```
python detect_plate.py --detect_model weights/plate_detect.pt  --rec_model weights/plate_rec_color.pth --video 2.mp4
```

视频文件为2.mp4  保存为result.mp4

## **车牌检测训练**

车牌检测训练链接如下：

[车牌检测训练](https://github.com/we0091234/Chinese_license_plate_detection_recognition/tree/main/readme)

## **车牌识别训练**

车牌识别训练链接如下：

[车牌识别训练](https://github.com/we0091234/crnn_plate_recognition)

#### **支持如下：**

- [X] 1.单行蓝牌
- [X] 2.单行黄牌
- [X] 3.新能源车牌
- [X] 4.白色警用车牌
- [X] 5.教练车牌
- [X] 6.武警车牌
- [X] 7.双层黄牌
- [X] 8.双层白牌
- [X] 9.使馆车牌
- [X] 10.港澳粤Z牌
- [X] 11.双层绿牌
- [X] 12.民航车牌

![Image ](image/README/test_1.jpg)

## 部署

1.[安卓NCNN](https://github.com/Ayers-github/Chinese-License-Plate-Recognition)

2.**onnx demo** 百度网盘： [k874](https://pan.baidu.com/s/1K3L3xubd6pXIreAydvUm4g)

```
python onnx_infer.py --detect_model weights/plate_detect.onnx  --rec_model weights/plate_rec_color.onnx  --image_path imgs --output result_onnx
```

3.**tensorrt** 部署见[tensorrt_plate](https://github.com/we0091234/chinese_plate_tensorrt)

4.**openvino demo** 版本2022.2

```
 python openvino_infer.py --detect_model weights/plate_detect.onnx --rec_model weights/plate_rec.onnx --image_path imgs --output result_openvino
```

## References

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
* [https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec](https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec)

## 联系

**有问题可以提issues 或者加qq群:871797331 询问**

![Image ](image/README/1.png)
