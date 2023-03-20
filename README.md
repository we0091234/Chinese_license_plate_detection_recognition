## What's New

**2022.12.04 车辆和车牌一起检测看这里[车辆系统](https://github.com/we0091234/Car_recognition)**

[yolov7 车牌检测+识别](https://github.com/we0091234/yolov7_plate)

[安卓NCNN](https://github.com/Ayers-github/Chinese-License-Plate-Recognition)

## **最全车牌识别算法，支持12种中文车牌类型**

**环境要求: python >=3.6  pytorch >=1.7**

#### **图片测试demo:**

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

1. **下载数据集：**  [datasets](https://pan.baidu.com/s/1xa6zvOGjU02j8_lqHGVf0A) 提取码：pi6c     数据从CCPD和CRPD数据集中选取并转换的
   数据集格式为yolo格式：

   ```
   label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
   ```

   关键点依次是（左上，右上，右下，左下）
   坐标都是经过归一化，x,y是中心点除以图片宽高，w,h是框的宽高除以图片宽高，ptx，pty是关键点坐标除以宽高
2. **修改 data/widerface.yaml    train和val路径,换成你的数据路径**

   ```
   train: /your/train/path #修改成你的路径
   val: /your/val/path     #修改成你的路径
   # number of classes
   nc: 2                 #这里用的是2分类，0 单层车牌 1 双层车牌

   # class names
   names: [ 'single','double']

   ```
3. **训练**

   ```
   python3 train.py --data data/widerface.yaml --cfg models/yolov5n-0.5.yaml --weights weights/plate_detect.pt --epoch 250
   ```

   结果存在run文件夹中
4. **检测模型  onnx export**
   检测模型导出onnx,需要安装onnx-sim  **[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)**

   ```
   1. python export.py --weights ./weights/plate_detect.pt --img 640 --batch 1
   2. onnxsim weights/plate_detect.onnx weights/plate_detect.onnx
   ```

   **训练好的模型进行检测**

   ```
   python detect_demo.py  --detect_model weights/plate_detect.pt
   ```

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

1. [安卓NCNN](https://github.com/Ayers-github/Chinese-License-Plate-Recognition)

2.**onnx demo** 百度网盘**： [k874](https://pan.baidu.com/s/1K3L3xubd6pXIreAydvUm4g)**

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
* [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

## 联系

**有问题可以提issues 或者加qq群:871797331 询问**
