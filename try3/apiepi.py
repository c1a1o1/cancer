'''
@author: botpi
'''
import tensorflow as tf

# convolutional

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def tf_queue(files):
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    name, data = reader.read(filename_queue)
        
    example = tf.parse_single_example(
        data,
        features = {
            'label': tf.FixedLenFeature([2], tf.int64),
            'vec': tf.FixedLenFeature([512, 100], tf.float32),
            'proj': tf.FixedLenFeature([100, 512], tf.float32),
            'mean': tf.FixedLenFeature([512], tf.float32),
        }
    )
    image = example['vec']
    #label = example['label']

    label = [tf.cast(example['label'], tf.float32)]
    # def f1(): return tf.constant([0., 1.])
    # def f2(): return tf.constant([1., 0.])
    # label = tf.cond(label[0]>0, f1, f2)

    x, y, z = tf.train.shuffle_batch(
        [image, label, name], batch_size = 1, 
        capacity = 1000,
        min_after_dequeue = 600)

    return x, y, z