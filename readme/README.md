### **车牌检测训练**

1. **下载数据集：**  [datasets](https://pan.baidu.com/s/1xa6zvOGjU02j8_lqHGVf0A) 提取码：pi6c     数据从CCPD和CRPD数据集中选取并转换的
   数据集格式为yolo格式：

   ```
   label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
   ```

   关键点依次是（左上，右上，右下，左下）
   坐标都是经过归一化，x,y是中心点除以图片宽高，w,h是框的宽高除以图片宽高，ptx，pty是关键点坐标除以宽高

   **自己的数据集**可以通过lablme 软件,create polygons标注车牌四个点即可，然后通过json2yolo.py 将数据集转为yolo格式，即可训练
2. **修改 data/widerface.yaml    train和val路径,换成你的数据路径**

   ```
   train: /your/train/path #修改成你的训练集路径
   val: /your/val/path     #修改成你的验证集路径
   # number of classes
   nc: 2                 #这里用的是2分类，0 单层车牌 1 双层车牌

   # class names
   names: [ 'single','double']

   ```
3. **训练**

   ```
   python3 train.py --data data/widerface.yaml --cfg models/yolov5n-0.5.yaml --weights weights/plate_detect.pt --epoch 120
   ```

   结果存在run文件夹中

### onnx export

1. 检测模型导出onnx,需要安装onnx-sim  **[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)**

   ```
   python export.py --weights ./weights/plate_detect.pt --img_size 640 --batch_size 1
   onnxsim weights/plate_detect.onnx weights/plate_detect.onnx
   ```
