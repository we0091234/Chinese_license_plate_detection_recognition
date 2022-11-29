from plate_recognition.plateNet import myNet_ocr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys

def cv_imread(path):  #可以读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value,std_value=(0.588,0.193)
def decodePlate(preds):
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    return newPreds

def image_processing(img,device):
    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def get_plate_result(img,device,model):
    input = image_processing(img,device)
    preds = model(input)
    # preds =preds.argmax(dim=2) #找出概率最大的那个字符
    # print(preds)
    preds=preds.view(-1).detach().cpu().numpy()
    newPreds=decodePlate(preds)
    plate=""
    for i in newPreds:
        plate+=plateName[i]
    # if not (plate[0] in plateName[1:44] ):
    #     return ""
    return plate

def init_model(device,model_path):
    # print( print(sys.path))
    # model_path ="plate_recognition/model/checkpoint_61_acc_0.9715.pth"
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']
    cfg=check_point['cfg']
    model_path = os.sep.join([sys.path[0],model_path])
    model = myNet_ocr(num_classes=78,export=True,cfg=cfg)
   
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

# model = init_model(device)
if __name__ == '__main__':

   image_path ="images/tmp2424.png"
   testPath = r"double_plate"
   fileList=[]
   allFilePath(testPath,fileList)
#    result = get_plate_result(image_path,device)
#    print(result)
   model = init_model(device)
   right=0
   begin = time.time()
   for imge_path in fileList:
        plate=get_plate_result(imge_path)
        plate_ori = imge_path.split('/')[-1].split('_')[0]
        # print(plate,"---",plate_ori)
        if(plate==plate_ori):

            right+=1
        else:
            print(plate_ori,"--->",plate,imge_path)
   end=time.time()
   print("sum:%d ,right:%d , accuracy: %f, time: %f"%(len(fileList),right,right/len(fileList),end-begin))
        
