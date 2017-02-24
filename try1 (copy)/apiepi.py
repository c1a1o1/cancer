'''
Created on Oct 24, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
import scipy.io
import sklearn.metrics as sk

def read_images(group):
    t = scipy.io.loadmat(group)
    t.pop("__globals__")
    t.pop("__version__")
    t.pop("__header__")
    
    
    pos = 0
    neg = 0
    nnan = 0
    images = []
    labels = []
    names = []
    for d in t:
        if not np.isnan(t[d]["corr"][0][0]).any():
            images.append(t[d]["corr"][0][0].ravel())
            if t[d]["label"][0][0][0]==1:
                labels.append([0, 1])
                pos +=1
            else:
                labels.append([1, 0])
                neg +=1
            names.append(d)
        else:
            nnan +=1
    print "data: pos, neg, nan", pos, neg, nnan
    return np.array(images), np.array(labels), names

def read_images_balanced(group, balance):
    images, labels, names = read_images(group)
    print images.shape
    
    images = images.tolist()
    labels = labels.tolist()
    
    images_pos = []
    labels_pos = []
    images_neg = []
    labels_neg = []    
    for i in xrange(len(labels)):
        if labels[i][0] == 0:
            images_pos.append(images[i])
            labels_pos.append(labels[i])
        else:
            images_neg.append(images[i])
            labels_neg.append(labels[i])
        
    images_bal_pos = []
    labels_bal_pos = []
    for i in xrange(balance):
        images_bal_pos += images_pos
        labels_bal_pos += labels_pos
    print "pos x %s" % balance, np.array(images_bal_pos).shape

    images = np.array(images_neg + images_bal_pos)
    labels = np.array(labels_neg + labels_bal_pos)
        
    return images, labels, names

def read_train_test(group, part):
    images, labels, names = read_images(group)
    
    images = images.tolist()
    labels = labels.tolist()
    
    images_pos = []
    labels_pos = []
    images_neg = []
    labels_neg = []    
    for i in xrange(len(labels)):
        if labels[i][0] == 0:
            images_pos.append(images[i])
            labels_pos.append(labels[i])
        else:
            images_neg.append(images[i])
            labels_neg.append(labels[i])
        
    npos = np.int(len(images_pos)*part)
    nneg = np.int(len(images_neg)*part)
    
    train_images = images_pos[:npos] + images_neg[:nneg]
    train_labels = labels_pos[:npos] + labels_neg[:nneg]
    test_images = images_pos[npos:] + images_neg[nneg:]
    test_labels = labels_pos[npos:] + labels_neg[nneg:]
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def F1c(labels, pred):
    tp = 0
    fp = 0
    fn = 0    
    for i in xrange(labels.shape[0]):
        if pred[i,1] == 1:
            if labels[i,1] == 1:
                tp += 1
            else:
                fp += 1
        elif labels[i,1] == 1:
            fn += 1        

    if tp + fp > 0:
        prec = tp / float(tp + fp)
    else:
        prec = 0
    
    if tp + fn > 0:
        rec =tp / float(tp + fn)
    else:
        rec = 0
    
    if rec + prec > 0:
        return 2 * prec * rec / (prec + rec), tp, fp, fn
    else:
        return 0, tp, fp, fn
    

class F1():
    def __init__(self):
        self.init()
    
    def init(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
    
    def take(self, cond, yval):
        if cond:
            if eval(yval) == 1:
                self.tp += 1
            else:
                self.fp += 1
        elif eval(yval) == 1:
            self.fn += 1        

    def calc(self):
        if self.tp + self.fp > 0:
            prec = self.tp / float(self.tp + self.fp)
        else:
            prec = 0
        
        if self.tp + self.fn > 0:
            rec = self.tp / float(self.tp + self.fn)
        else:
            rec = 0
        
        if rec + prec > 0:
            return 2 * prec * rec / (prec + rec)
        else:
            return 0

def auc(labels, prob):
    return sk.roc_auc_score(labels[:,1], prob[:,1])

# convolutional

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
