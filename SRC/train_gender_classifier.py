# 训练结果csv记录日志  模型checkpoint 监测值不再改善时--回调函数终止训练
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping
# 当评价指标不再提升时，较小学习率
from keras.callbacks import ReduceLROnPlateau
# mini_XCEPTION 模型
from src.models.cnn import mini_XCEPTION
# 类用于加载imdb性别分类数据集
from src.utils.datasets import DataManager
#图像发生器与饱和度，亮度，照明，对比度，水平翻转和垂直翻转转换
from src.utils.data_augmentation import ImageGenerator
# 训练数据拆分
from src.utils.datasets import split_imdb_data

# 模型参数
batch_size = 32
num_epoch = 1000
validation_split = 0.2 # 训练集80% 验证集20%
do_random_crop = False # 图像预处理---随机剪裁
patience = 100
num_classes = 2 # 二分类：man/woman
dataset_name = 'imdb'
input_shape = (64,64,1) # 输入image大小
if input_shape[2] == 1:
    grayscale = True

# 图像、日志、训练模型保存路径
image_path = '../datasets/imdb_crop/'
log_file_path = '../trained_models/gender_models/gender_training.log'
trained_models_path = '../trained_models/gender_models/gender_mini_XCEPTION'

# data loader
data_loader = DataManager(dataset_name) # imdb数据集
ground_truth_data = data_loader.get_data()
train_keys,val_keys = split_imdb_data(ground_truth_data,validation_split)

print('The Number of training samples:',len(train_keys))
print('The Number of validation samples:',len(val_keys))
# 图像预处理
image_generator = ImageGenerator(ground_truth_data,batch_size,input_shape[:2],
                                 train_keys,val_keys,None,path_prefix=image_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,do_random_crop=do_random_crop)

# 模型参数
model = mini_XCEPTION(input_shape,num_classes)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# 模型回调函数

# 终止训练       1: 需要监视的变量  2：如果loss相比上1个epoch没有下降，则经过patience个epoch后停止训练
early_stop = EarlyStopping(monitor='val_loss',patience=patience)
# 当评价指标不在提升时，减少学习率
# 1：监视的变量 2：每次减少学习率的因子*lr 3:patience个epoch后acc不提升 主动减少学习率
reduce_lr = ReduceLROnPlateau(monitor='vai_loss',factor=0.1,patience=int(patience/2),verbose=1)
# 1:保存的文件名 2：False--总是覆盖csv文件
csv_logger = CSVLogger(log_file_path,append=False)
model_name = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
# 在每个epoch后保存模型到filepath
# 1:filepath 保存的路径 2：监视的变量 3：信息展示 4：True时--只保存在验证集上性能最好的模型
# 5：True-只保存模型权重 / False-保存整个模型(包括模型结构，配置信息)
model_checkpoint = ModelCheckpoint(filepath=model_name,monitor='vol_loss',
                                   verbose=1,save_best_only=True,
                                   save_weights_only=False)


# train model:
# 回调函数【】
callbacks = [model_checkpoint,csv_logger,early_stop,reduce_lr]
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epoch,verbose=1,callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))
