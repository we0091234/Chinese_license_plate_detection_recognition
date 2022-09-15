from plate_recognition.plateNet import myNet
# from plateNet import myNet
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def init_model():

    model_paramas = torch.load(r"plate_recognition/0.9973404255319149_epoth_1_model.pth.tar",map_location=device)
    # state_dict=model_paramas
    state_dict = model_paramas['state_dict']
    cfg=model_paramas['cfg']
    # cfg=[32, 'M', 64, 'M', 72, 'M', 104, 'M', 52]
    model=myNet(num_classes=2,cfg=cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def load_mean():
    MEAN_NPY = r'plate_recognition/plate.npy'
    mean_npy = np.load(MEAN_NPY)
    mean = mean_npy.mean(1).mean(1)
    return mean
def img_process(img,mean):
    img=cv2.resize(img,(64,64))
    img = img-mean
    img=np.transpose(img,(2,0,1))
    img=img.reshape(-1,3,64,64)
    img=torch.from_numpy(img).float()
  
    img=img.to(device)
    return img

def get_sample_label(img_path):
     imageName = img_path.split("/")[-1]
     pos1=imageName.rfind(".")
     pos2=imageName.rfind("-")
     imageLabel = imageName[pos2+1:pos1]
     return imageLabel

def get_sin_dou_plate(img):
    mean = load_mean()
    img = img_process(img,mean)
    model.eval()
    result = model(img)
    result = torch.softmax(result,dim=1)
    result=torch.argmax(result,dim=1).item()
    return result

model = init_model()

if __name__ == "__main__":
    img_path =r"/mnt/Gpan/Mydata/pytorchPorject/imageClass/myNetTraing/_dataSets/single_double/val"
    file_list = []
    allFilePath(img_path,file_list)
    
    # model = init_model()
   
    img_path = r"_dataSets/single_double/val/1/double_plate_1/plate(0)-1.jpg"
    right = 0
    for img_path in file_list:
        img = cv_imread(img_path)
        
       
        # img = img.transpose(2,0,1)
        label=get_sample_label(img_path)
        result = get_sin_dou_plate(img)
        if result==int(label):
            right+=1
        print(img_path,result,label)
        # print(result.shape)
    print(f"sum={len(file_list)},right={right},ratio={right/len(file_list)}")

   
    



    