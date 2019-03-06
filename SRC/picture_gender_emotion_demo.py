# 人脸面部表情 以及 性别识别分类
import cv2
import numpy as np

from statistics import mode
from keras.models import load_model
from src.utils.datasets import get_labels # 拿到标签{emotion/性别}
from src.utils.inference import detect_faces # 人脸检测
from src.utils.inference import draw_text # 写文字
from src.utils.inference import draw_bounding_box # 绘框
from src.utils.inference import apply_offsets # 坐标设置
from src.utils.inference import load_detection_model # 加载人脸检测分类器
from src.utils.preprocessor import preprocess_input # 输入图像预处理

# 面部表情及其性别识别分类
# parameters for loading data and images
# 人脸检测分类器模型
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# 人脸情绪识别分类模型
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
# 人物性别识别分类模型
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'

# emotion_labels {0-6}
emotion_labels = get_labels('fer2013')
# gender_labels {0-1}
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX # 定义字体 -- 正常尺寸的sans-serif字体
# 人脸检测框的超参数
frame_window = 10
emotion_offsets = (20,40) # emotion坐标
gender_offsets = (30,60) # gender坐标

# 加载模型
face_detection = load_detection_model(detection_model_path) # 人脸检测分类器
emotion_classifier = load_model(emotion_model_path,compile=False) # 人脸情趣模型
gender_classifier = load_model(gender_model_path,compile=False) # 性别分类模型

# input_model_shape()
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# 模型列表
gender_window = []
emotion_window = []

# 开始采集视频图像
cv2.namedWindow('emotion_frame')
video_capture = cv2.VideoCapture(0) # 摄像头
while True:
    # 抓一帧图
    # bgr_image = video_capture.read()[1] # bgr图
    bgr_image = cv2.imread('../images/test_image.jpg')
    gray_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2GRAY) #灰度图
    rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB) # 彩色图
    faces = detect_faces(face_detection,gray_image) #检测到的人脸

    for face_coordinates in faces:
        # 对每一个人脸，分别找到 gender 和 emotion 坐标
        x1,x2,y1,y2 = apply_offsets(face_coordinates,gender_offsets)
        rgb_face = rgb_image[y1:y2,x1:x2]

        x1,x2,y1,y2 = apply_offsets(face_coordinates,emotion_offsets)
        gray_face = gray_image[y1:y2,x1:x2]

        # 调整大小
        try:
            rgb_face = cv2.resize(rgb_face,(gender_target_size))
            gray_face = cv2.resize(gray_face,(emotion_target_size))
        except:
            continue
        ##########  emotion ##########
        gray_face = preprocess_input(gray_face,False) # 预处理 (64,64)
        # print('gray_face0:', gray_face.shape)
        gray_face = np.expand_dims(gray_face,0) #(1,64,64)
        # print('gray_face1:', gray_face.shape)
        gray_face = np.expand_dims(gray_face,-1)#(1,64,64,1)
        # print('gray_face2:',gray_face.shape)

        # emotion_label
        # predict:每一个label的得分值
        emotion_prediction = emotion_classifier.predict(gray_face)
        # print('pre_label:',emotion_prediction)
        # argmax取最大概率
        emotion_label_arg = np.argmax(emotion_prediction) #最大值的索引
        # print('emotion_label_arg',emotion_label_arg)
        # emotion-text
        emotion_text = emotion_labels[emotion_label_arg] # 由索引拿到真实标签
        # print('emotion_text',emotion_text)
        emotion_window.append(emotion_text)

        ##########  gender ##########
        rgb_face = np.expand_dims(rgb_face,0)
        print('rgb_face:',rgb_face.shape) # (1,48,48,1)
        rgb_face = preprocess_input(rgb_face,False) # rgb图片预处理
        # gender_predict
        gender_prediction = gender_classifier.predict(rgb_face)
        # print('gender_prediction',gender_prediction)
        gender_label_arg = np.argmax(gender_prediction) # 最大值索引
        # print('gender_label_arg:',gender_label_arg)
        # gender_text:
        gender_text = gender_labels[gender_label_arg] # 由索引拿到真实label
        # print('gender-text:',gender_text)
        gender_window.append(gender_text)

        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0,255,0)
        else:
            color = (255,0,0)

        # 绘制人脸框
        draw_bounding_box(face_coordinates,rgb_image,color)
        # 绘制emotion/gender属性文字
        draw_text(face_coordinates,rgb_image,gender_mode,color,0,-20,1,1)
        draw_text(face_coordinates,rgb_image,emotion_mode,color,0,-45,1,1)
    bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR) # bgr图
    cv2.imshow('emotion_frame',bgr_image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break



