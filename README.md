## **车牌识别算法**
**环境要求: python >=3.6  pytorch >=1.7**

## 数据集准备
D:/datasets/images用于存放图片，/train存放训练图片，/val存放验证图片，/test存放测试图片。  
D:/datasets/labels用于存放标签，格式同上。  
注意：需要有记录图片标注信息的Excel表，具体格式参照项目文件CLPD.xlsx  
在data_process.py中更改excel_dir的地址并且运行，以生成标签。  

## 如何运行？

运行detect_plate.py 或者运行如下命令行
```
python detect_plate.py --detect_model weights/plate_detect.pt  --rec_model weights/plate_rec_color.pth --image_path imgs --output result
```
就能对/test文件夹中的图片进行检测并将结果输出到/result文件夹

## 车牌检测定位模型训练
```
python det_train.py
```

## 车牌识别数据集准备
图片命名为：**车牌号_序号.jpg**  
如何得到小图（就是用来训练的图片）  
运行detect_plate.py，选择在datasets/det_results生成小图并重命名。  
然后放入datasets/rec_train和datasets/rec_val，执行如下命令，给数据集打上标签,生成train.txt和val.txt
```
python rec_plateLabel.py --image_path D:/datasets/images/rec_train/ --label_file datasets/train.txt
python rec_plateLabel.py --image_path D:/datasets/images/rec_val/ --label_file datasets/val.txt
```
注：train.txt，val.txt路径可以在lib/config/rec_data.yaml中更改
数据格式如下：
```
D:/datasets/images/rec_val/冀BAJ731_3.jpg 5 53 52 60 49 45 43 
   ```
   


## 车牌识别模型训练
默认使用自己搭建的模型--myNet进行训练，如果要训练其他模型，使用格式如：python rec_train.py  --model rec_myNet  
可供选择的模型参考项目文件夹/plate_recognition，其中，训练最完善（项目最终采用）的模型为--model rec_cnn  
训练结果（检查点及权重）保存在output文件夹，其中，最好的训练结果（项目最终采用）保存在/weights文件夹
```
python rec_train.py
```

## 导出onnx
```
python export.py --weights saved_model/best.pth --save_path saved_model/best.onnx  --simplify
```
## onnx 推理
```
python onnx_infer.py --onnx_file saved_model/best.onnx  --image_path images/test.jpg
```

## 双层车牌

双层车牌这里采用拼接成单层车牌的方式：

python:

```
def get_split_merge(img):
    h,w,c = img.shape
    img_upper = img[0:int(5/12*h),:]
    img_lower = img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
    new_img = np.hstack((img_upper,img_lower))
    return new_img
```

c++:

```
cv::Mat get_split_merge(cv::Mat &img)   //双层车牌 分割 拼接
{
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));
    return out;
}
```

## 数据增强
```
cd Text-Image-Augmentation-python-master

python demo1.py --src_path /mnt/Gu/trainData/test_aug --dst_path /mnt/Gu/trainData/result_aug/
```

src_path 是数据路径， dst_path是保存的数据路径

**然后把两份数据放到一起进行训练，效果会好很多！**



