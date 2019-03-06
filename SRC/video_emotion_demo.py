from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from src.utils.datasets import get_labels # 拿到数据标签
from src.utils.inference import detect_faces # 人脸框
from src.utils.inference import draw_text # 写入文本
from src.utils.inference import draw_bounding_box # 画人脸框
from src.utils.inference import apply_offsets #
from src.utils.inference import load_detection_model # 加载模型
from src.utils.preprocessor import preprocess_input #
# 人脸检测分类器
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# 面部表情识别
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# 超参数--用于包围框的
frame_window = 10
emotion_offsets = (20,40)

# load_model
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path,compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []

cv2.namedWindow('Danbin_face')
video_capture = cv2.VideoCapture(0) # 摄像头
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2GRAY) # 灰度图
    rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)   # 彩色图
    # 人脸
    faces = detect_faces(face_detection,gray_image)

    for face_coordinates in faces:
        x1,x2,y1,y2 = apply_offsets(face_coordinates,emotion_offsets)
        gray_face = gray_image[y1:y2,x1:x2]
        try:
            gray_face = cv2.resize(gray_face,(emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face,True)
        gray_face = np.expand_dims(gray_face,0)
        gray_face = np.expand_dims(gray_face,-1)

        # 人脸表情预测 -- 输入灰度图 ---》 7个label不同分数
        emotion_prediction = emotion_classifier.predict(gray_face) #
        # print('emotion_predicted:',emotion_prediction)

        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)

        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255,0,0)) # red
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0,0,255)) # blue
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255,255,0)) # yellow
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0,255,255)) # 青色
        else:
            color = emotion_probability * np.asarray((0,255,0)) # 绿色
        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates,rgb_image,color)
        draw_text(face_coordinates,rgb_image,emotion_mode,color,0,-45,1,1)
    bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR)
    cv2.imshow('Danbin_face',bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



