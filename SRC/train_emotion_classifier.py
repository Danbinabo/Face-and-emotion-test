# 面部情感识别 ---- train
# 下载训练数据集 tar-xzf fer2013.tar
# 训练结果csv记录日志  模型checkpoint 监测值不再改善时--回调函数终止训练
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping
# 当评价指标不再提升时，较小学习率
from keras.callbacks import ReduceLROnPlateau
# 图片预处理
from keras.preprocessing.image import ImageDataGenerator

# mini_XCEPTION 模型
from src.models.cnn import mini_XCEPTION
# 类用于加载fer2013情绪分类数据集 或 imdb性别分类数据集
from src.utils.datasets import DataManager
# 数据集拆分----训练/验证/测试
from src.utils.datasets import split_data

from src.utils.preprocessor import preprocess_input


# 模型参数

batch_size = 32
num_epochs = 10000
input_shape = (64,64,1) # 输入图像大小 64 * 64 的灰度图
validation_split = 0.2 # 80%训练 20%验证
verbose = 1
num_classes = 7 # labels
patience = 50

# 模型保存路径
base_path = '../trained_models/emotion_models/'

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True)

# 模型参数
# mini_XCEPTION模型
model = mini_XCEPTION(input_shape,num_classes)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

datasets = ['fer2013']

for dataset_name in datasets:
    print('Training dataset:',dataset_name)
    log_file_path = base_path + dataset_name + '_emotation_training.log'
    csv_logger = CSVLogger(log_file_path,append=False)
    early_stop = EarlyStopping('val_loss',patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss',fator=0.1,patience=int(patience/4),verbose=1)
    train_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = train_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names,'val_loss',verbose=1,save_best_only=True)

    callbacks = [model_checkpoint,csv_logger,early_stop,reduce_lr]

    # load_dataset:
    data_loader = DataManager(dataset_name,image_size=input_shape[:2])
    faces,emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples,num_classes = emotions.shape
    train_data,val_data = split_data(faces,emotions,validation_split)
    train_faces,train_emotions = train_data
    model.fit_generator(data_generator.flow(train_faces,train_emotions,batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs,verbose=1,callbacks=callbacks,
                        validation_data=val_data)



















