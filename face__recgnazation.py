import argparse
from utils.utils import generate_bbox, py_nms, convert_to_square
from utils.utils import pad, calibrate_box, processed_image
from arc_face import *
from torch.nn import DataParallel
import os
import pyttsx3
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='infer_models_weights',      help='PNet、RNet、ONet三个模型文件存在的文件夹路径')
args = parser.parse_args()
device = torch.device("cuda")
# 获取P模型
pnet = torch.jit.load(os.path.join(args.model_path, 'PNet.pth'))
pnet.to(device)
softmax_p = torch.nn.Softmax(dim=0)
pnet.eval()

# 获取R模型
rnet = torch.jit.load(os.path.join(args.model_path, 'RNet.pth'))
rnet.to(device)
softmax_r = torch.nn.Softmax(dim=-1)
rnet.eval()

# 获取R模型
onet = torch.jit.load(os.path.join(args.model_path, 'ONet.pth'))
onet.to(device)
softmax_o = torch.nn.Softmax(dim=-1)
onet.eval()
# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()
# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()
# 使用ONet模型预测
def predict_onet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32, device=device)
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = onet(infer_data)
    cls_prob = softmax_o(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()
# 获取PNet网络输出结果
def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape
        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T
    del all_boxes
    return boxes_c
# 获取RNet网络输出结果
def detect_rnet(im, dets, thresh):
    h, w, c = im.shape
    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None
    keep = py_nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c
# 获取ONet模型预测结果
def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) / 128
        cropped_ims[i, :, :, :] = img
    cls_scores, reg, landmark = predict_onet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)
    keep = py_nms(boxes_c, 0.6, mode='Minimum')
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark
# 人脸检测
def face_detective(im):
    # 调用第一个模型预测
    boxes_c = detect_pnet(im, 20, 0.79, 0.9)
    if boxes_c is None:
        return None, None
    # 调用第二个模型预测
    boxes_c = detect_rnet(im, boxes_c, 0.6)
    if boxes_c is None:
        return None, None
    # 调用第三个模型预测
    boxes_c, landmark = detect_onet(im, boxes_c, 0.7)
    if boxes_c is None:
        return None, None
    return boxes_c, landmark
def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = cv2.resize(image,(128,128))
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image
def get_featuresdict(model, dir):
    list1 = os.listdir(dir)
    person_dict = {}
    for i,each in enumerate(list1):
        image = load_image(f"pic/{each}")
        data = torch.from_numpy(image)
        data = data.to(torch.device("cuda"))
        output = model(data)  # 获取特征
        output = output.data.cpu().numpy()
        # 获取不重复图片 并分组
        fe_1 = output[0]
        fe_2 = output[1]
        feature = np.hstack((fe_1, fe_2))
        person_dict[each] = feature
    return person_dict
def cosin_metric(x1, x2):
    #计算余弦距离
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
def draw_face(img, boxes_c,label):
    corpbbox = [int(boxes_c[0]), int(boxes_c[1]), int(boxes_c[2]), int(boxes_c[3])]
    # 画人脸框
    cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                  (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)
    # 填写识别名字
    cv2.putText(img, label,
                (corpbbox[0], corpbbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
# 人脸识别
def face_recognition(img):
    img0 = img
    uu=0

    boxes_c, landmarks = face_detective(img)

    label = "none"
    # engine = pyttsx3.init()


    if boxes_c is not None:
        for det in boxes_c:
            det[det < uu] =uu  # 坐标会有负值，一律给0
            det[1] =det[1]- uu
            print(det)
            face_img = img[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
            face_img = cv2.resize(face_img, (128, 128))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = np.dstack((face_img, np.fliplr(face_img)))
            face_img = face_img.transpose((2, 0, 1))
            face_img = face_img[:, np.newaxis, :, :]
            face_img = face_img.astype(np.float32, copy=False)
            face_img -= 127.5
            face_img /= 127.5
            face_data = torch.from_numpy(face_img)
            face_data = face_data.to(device)
            _output = arcface_model(face_data)  # 获取特征
            _output = _output.data.cpu().numpy()
            fe_1 = _output[0]
            fe_2 = _output[1]
            _feature = np.hstack((fe_1, fe_2))
            label = "none"
            list3 = os.listdir(dir)
            max_f = 0
            t = 0
            for each in list3:
                t = cosin_metric(features[each], _feature)
                if t > max_f:
                    max_f = t
                    max_n = each
                if (max_f > 0.9):#门限阈值
                    label = max_n[:-4]
            max_fafer=max_f
            print(max_f)
            # engine.say(round(max_f,2))
            # engine.runAndWait()
            draw_face(img0, det, label)

    return (img0,label)
# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
if __name__ == '__main__':
    save_img = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir = 'pic'#图片库
    arcface_model = resnet_face18(False)
    out='output'#输出库
    sum=0
    arcface_model = DataParallel(arcface_model)
    arcface_model.load_state_dict(torch.load(r'infer_models_weights/resnet18_110.pth'), strict=False)
    arcface_model.to(device).eval()
    features = get_featuresdict(arcface_model, dir)
    source=''#空为摄像头采集，可填文件夹
    if source =='':
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if ret:
                img0,label = face_recognition(img)#识别人脸
                cv2.imshow('face rec', img0)
                cv2.waitKey(100)
    else:
        n=0
        m=0
        error=0
        for filename in os.listdir(source):
            print('当前帧:' +str(m+1))
            img = cv2.imread(source+'/'+filename)
            img0,label=face_recognition(img)
            if label== "none":
                print('当前空标帧:'+str(n))
                img0 = rotate_bound_white_bg(img, 15)#识别识别旋转15°
                img0, label= face_recognition(img0)
                if label== "none":
                    print('第二次空标帧:'+str(n))
                    img0 = rotate_bound_white_bg(img, -15)#识别识别反向旋转15°
                    img0, label = face_recognition(img0)
            if label==source:
                n=n+1
                print('当前有效帧' + str(n))
            elif label!="none":
                error=error+1
                print('当前错误帧' + str(error))
            cv2.imshow('face rec', img0)
            cv2.waitKey(100)
            m = m + 1
            cv2.imwrite(out + '/' + str(m) + '.jpg', img0)
        print('识别成功率' +str( n/m))
        print('张冠李戴率' + str(error/m))
