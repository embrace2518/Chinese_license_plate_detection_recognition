# -*- coding: UTF-8 -*-
import argparse
import time
import os
import cv2
import torch
import copy
import numpy as np
import data_process
import myshow
from lib.utils import img_process
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']

DET_PATH = 'D:/datasets/images/det_result'
SAVE_PATH = 'D:/datasets/images/result'
IMGS_PATH = 'D:/datasets/images/test'


# 获取车牌坐标以及四个角点坐标并获取车牌号
def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model,
                           is_color=False):
    h, w, c = img.shape
    result_dict = {}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])
    class_label = int(class_num)  # 车牌的的类型0代表单牌，1代表双层车牌
    roi_img = img_process.four_point_transform(img, landmarks_np)  # 透视变换得到车牌小图
    if class_label:
        roi_img = get_split_merge(roi_img)
    if not is_color:
        plate_number, rec_prob = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)
    else:
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model,
                                                                           is_color=is_color)
    result_dict['rect'] = rect  # 车牌roi区域
    result_dict['detect_conf'] = conf  # 检测区域得分
    result_dict['landmarks'] = landmarks_np.tolist()  # 车牌角点坐标
    result_dict['plate_no'] = plate_number  # 车牌号
    result_dict['rec_conf'] = rec_prob  # 每个字符的概率
    result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
    result_dict['plate_color'] = ""
    if is_color:
        result_dict['plate_color'] = plate_color  # 车牌颜色
        result_dict['color_conf'] = color_conf  # 颜色得分
    result_dict['plate_type'] = class_label  # 单双层 0单层 1双层

    return result_dict, roi_img


# 获取车牌信息
def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, is_color=False):
    # Load model
    # img_size = opt_img_size
    conf_thres = 0.3  # 得分阈值
    iou_thres = 0.5  # nms的iou值
    dict_list = []
    roi_img = None
    img0 = copy.deepcopy(orgimg)  # 深拷贝
    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]  # 检测前处理，图片长宽变为32倍数，比如变为640X640
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416  图片的BGR排列转为RGB,然后将图片的H,W,C排列变为C,H,W排列
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = non_max_suppression_face(model(img)[0], conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = img_process.scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()  # NumPy无法直接处理GPU张量
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict, roi_img = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device,
                                                              plate_rec_model,
                                                              is_color=is_color)
                dict_list.append(result_dict)
    return dict_list, roi_img


# 车牌结果画出来
def draw_result(orgimg, dict_list, is_color=False):
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result_p = result['plate_no']
        if result['plate_type'] == 0:  # 单层
            result_p += " " + result['plate_color']
        else:  # 双层
            result_p += " " + result['plate_color'] + "双层"
        result_str += result_p + " "
        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 获得字体的大小
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:  # 防止显示的文字越界
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
        orgimg = cv2.rectangle(orgimg, (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                               (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]),
                               (255, 255, 255), cv2.FILLED)  # 画文字框,背景白色

        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                   (0, 0, 0), int(result['roi_height'] * 0.6))
            # orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)

    print(result_str)
    return orgimg


# 识别单张图片
def process_single_image(img_path):
    global count, count_right, time_all
    print(count, img_path, end=" ")
    time_b = time.time()
    img = cv_imread(img_path)
    # 在图片处理前添加检查
    if img is None:
        return
    if (img.shape[-1] == 4  # 如果是4通道
            or img.shape[2] == 4  # 灰度图
            or len(img.shape) == 2):  # RGBA图
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为BGR

    dict_list, det_img = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                  is_color=opt.is_color)  # 检测以及识别车牌
    #  生成小图
    img_name = os.path.basename(img_path)
    if make_roi_img == '1' and det_img is not None:
        det_img = cv2.resize(det_img, (168, 48), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(opt.det_output, img_name), det_img)
    ori_img = draw_result(img, dict_list)
    save_img_path = os.path.join(save_path, img_name)
    cv2.imwrite(save_img_path, ori_img)
    time_all += time.time() - time_b
    count += 1
    for result in dict_list:
        img_label = result['plate_no']
        if right_img(img_name, img_label):
            count_right += 1
        else:
            print('识别错误的图片：', img_name)

def right_img(img_name, img_label):
    labels = data_process.get_name_label()
    return img_label == labels[img_name]


def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)  # 帧速率
        FrameNumber = capture.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return int(rate), int(FrameNumber), int(duration)


if __name__ == '__main__':
    make_roi_img = input("是否生成小图(生成1/不生成0）")
    show_img = input("是否展示(展示1/不展示0）")

    parser = argparse.ArgumentParser()  # 创建解析器
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt',
                        help='model.pt path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌识别+颜色识别模型
    parser.add_argument('--is_color', type=bool, default=True, help='plate color')  # 是否识别颜色
    parser.add_argument('--img_path', type=str, default=IMGS_PATH, help='source')  # 图片路径
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  # 网络输入图片大小
    parser.add_argument('--output', type=str, default=SAVE_PATH, help='source')  # 图片结果保存的位置
    parser.add_argument('--det_output', type=str, default=DET_PATH, help='source')  # 检测结果保存的位置
    parser.add_argument('--video', type=str, default='', help='source')  # 视频的路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu还是cpu进行识别
    opt = parser.parse_args()  # 解析命令行参数
    print(opt)
    save_path = opt.output
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    detect_model = attempt_load(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)
    print("detect params: %.2fM,rec params: %.2fM" % (
        sum(p.numel() for p in detect_model.parameters()) / 1e6,
        sum(p.numel() for p in plate_rec_model.parameters()) / 1e6))

    count = 0
    count_right = 0
    time_all = 0
    time_begin = time.time()

    if not opt.video:
        if not os.path.isfile(opt.img_path):
            file_list = []
            allFilePath(opt.img_path, file_list)  # 读取所有图片路径
            for img_path in file_list:
                process_single_image(img_path)
            print(f"Accuracy is {count_right / count * 100}%\n"
                  f"sumTime time is {time.time() - time_begin} s,average pic time is {time_all / len(file_list)} s")
        else:
            process_single_image(opt.img_path)

    else:
        video_name = opt.video
        capture = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))  # 写入视频
        frame_count = 0
        fps_all = 0
        rate, FrameNumber, duration = get_second(capture)
        if capture.isOpened():
            while True:
                t1 = cv2.getTickCount()
                frame_count += 1
                print(f"第{frame_count} 帧", end=" ")
                ret, img = capture.read()
                if not ret:
                    break
                img0 = copy.deepcopy(img)
                dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                     is_color=opt.is_color)
                ori_img = draw_result(img, dict_list)
                t2 = cv2.getTickCount()
                infer_time = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / infer_time
                fps_all += fps
                str_fps = f'fps:{fps:.4f}'
                cv2.putText(ori_img, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(ori_img)
        else:
            print("失败")
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"all frame is {frame_count},average fps is {fps_all / frame_count} fps")

    if show_img == '1':
        myshow.show_result(SAVE_PATH)
