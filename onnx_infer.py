import onnxruntime
import numpy as np
import cv2
import copy
import os

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
def my_letter_box(img,size=(640,640)):
    h,w,c = img.shape
    r = min(size[0]/h,size[1]/w)
    new_h,new_w = int(h*r),int(w*r)
    top = int((size[0]-new_h)/2)
    left = int((size[1]-new_w)/2)
    
    bottom = size[0]-new_h-top
    right = size[1]-new_w-left
    img_resize = cv2.resize(img,(new_w,new_h))
    img = cv2.copyMakeBorder(img_resize,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def xywh2xyxy(boxes):
    xywh =copy.deepcopy(boxes)
    xywh[:,0]=boxes[:,0]-boxes[:,2]/2
    xywh[:,1]=boxes[:,1]-boxes[:,3]/2
    xywh[:,2]=boxes[:,0]+boxes[:,2]/2
    xywh[:,3]=boxes[:,1]+boxes[:,3]/2
    return xywh

def my_nms(boxes,iou_thresh):
    index = np.argsort(boxes[:,4])[::-1]
    keep = []
    while index.size >0:
        i = index[0]
        keep.append(i)
        x1=np.maximum(boxes[i,0],boxes[index[1:],0])
        y1=np.maximum(boxes[i,1],boxes[index[1:],1])
        x2=np.minimum(boxes[i,2],boxes[index[1:],2])
        y2=np.minimum(boxes[i,3],boxes[index[1:],3])
        
        w = np.maximum(0,x2-x1)
        h = np.maximum(0,y2-y1)

        inter_area = w*h
        union_area = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])+(boxes[index[1:],2]-boxes[index[1:],0])*(boxes[index[1:],3]-boxes[index[1:],1])
        iou = inter_area/(union_area-inter_area)
        idx = np.where(iou<=iou_thresh)[0]
        index = index[idx+1]
    return keep

def restore_box(boxes,r,left,top):
    boxes[:,[0,2,5,7,9,11]]-=left
    boxes[:,[1,3,6,8,10,12]]-=top

    boxes[:,[0,2,5,7,9,11]]/=r
    boxes[:,[1,3,6,8,10,12]]/=r
    return boxes
    

def post_precessing(dets,r,left,top,conf_thresh=0.3,iou_thresh=0.5):
    choice = dets[:,:,4]>conf_thresh
    dets=dets[choice]
    dets[:,13:15]*=dets[:,4:5]
    box = dets[:,:4]
    boxes = xywh2xyxy(box)
    score= np.max(dets[:,13:15],axis=-1,keepdims=True)
    index = np.argmax(dets[:,13:15],axis=-1).reshape(-1,1)
    output = np.concatenate((boxes,score,dets[:,5:13],index),axis=1) 
    reserve_=my_nms(output,iou_thresh) 
    output=output[reserve_]
    output = restore_box(output,r,left,top)
    return output


if __name__ == "__main__":
    file_list = []
    allFilePath(r"imgs",file_list)
    save_path = "result1"
    providers =  ['CPUExecutionProvider']
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    session = onnxruntime.InferenceSession(r"weights/plate_detect_sim.onnx", providers=providers)
    for pic_ in file_list:
        img=cv2.imread(pic_)
        img0 = copy.deepcopy(img)
        img,r,left,top=my_letter_box(img)
        # cv2.imwrite("1.jpg",img)
        img =img[:,:,::-1].transpose(2,0,1).copy().astype(np.float32)
        img=img/255
        img=img.reshape(1,*img.shape)
        # print(img.shape)
        y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img})[0]
        outputs = post_precessing(y_onnx,r,left,top)
        for output in outputs:
            output = output.tolist()
            rect=output[:4]
            land_marks=output[5:13]
            cv2.rectangle(img0,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(0,255,255),thickness=3)
            for i in range(4):
                cv2.circle(img0,(int(land_marks[2*i]),int(land_marks[2*i+1])),5,clors[i],-1)
        #     cv2.imshow("haha",img0)
        # cv2.waitKey(0)
        print(pic_)
        img_name = os.path.basename(pic_)
        new_pic_path = os.path.join(save_path,img_name)
        cv2.imwrite(new_pic_path,img0)


        