import os
import tensorflow as tf


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def read_and_decode(filename, reshape_dims):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # uint8(0~255) 타입을 float32(0~1) 타입으로 변환
    
    # 종횡비 유지하지 않음
    return tf.image.resize(img, reshape_dims)


def decode_csv(csv_row):
    record_defaults = ['path', 'label']
    path_file, label = tf.io.decode_csv(csv_row, record_defaults)
    img = read_and_decode(path_file, [IMG_HEIGHT, IMG_WIDTH])
    label = tf.argmax(tf.math.equal(CLASS_NAMES, label))
    
    return img, label


def get_datasets(batch_size=10):
    path_data = 'data/5-flowers'
    train_dataset = tf.data.TextLineDataset(
        os.path.join(path_data, 'train.csv')
    ).skip(1).map(decode_csv).batch(batch_size)
    eval_dataset = tf.data.TextLineDataset(
        os.path.join(path_data, 'test.csv')
    ).skip(1).map(decode_csv).batch(batch_size)
    
    return train_dataset, eval_dataset