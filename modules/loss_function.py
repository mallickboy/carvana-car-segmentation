import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

@tf.keras.utils.register_keras_serializable()
def iou_coeff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f) + K.epsilon()
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection + K.epsilon()

    return intersection / union

def iou_loss(y_true, y_pred):
    return 1 - iou_coeff(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def dice_coeff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    total_area = K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()

    return (2 * intersection + K.epsilon()) / total_area

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    
    return 0.3 * bce + 0.7 * dice

@tf.keras.utils.register_keras_serializable()
def pixel_accuracy(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    correct_pixels = K.sum(tf.cast(K.equal(y_true, y_pred), tf.float32))
    total_pixels = K.prod(K.shape(y_true))

    return correct_pixels / tf.cast(total_pixels, tf.float32)

@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

@tf.keras.utils.register_keras_serializable()
def bce_iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)

    return 0.5 * bce + 0.5 * iou


## explicitely mentionaing as keras is unable to find
get_custom_objects()['bce_iou_loss'] = bce_iou_loss
get_custom_objects()['bce_dice_loss'] = bce_dice_loss

get_custom_objects()['iou_coeff'] = iou_coeff
get_custom_objects()['dice_coeff'] = dice_coeff
get_custom_objects()['f1_score'] = f1_score
get_custom_objects()['pixel_accuracy'] = pixel_accuracy