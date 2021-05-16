import tensorflow as tf
import tensorflow.keras.backend as K


def interval_overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2

    if x3 < x1:
        return 0 if x4 < x1 else (min(x2, x4) - x1)
    else:
        return 0 if x2 < x3 else (min(x2, x4) - x3)


def intersection_over_union(box1, box2):
    print(box1.constant)
    intersect_w = interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = interval_overlap([box1[..., 1], box1[..., 3]], [box2[..., 1], box2[..., 3]])
    intersect_area = intersect_h * intersect_w

    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]

    union_area = w1 * h1 + w2*h2 - intersect_area

    return float(intersect_area) / union_area


# def intersection_over_union( target_boxes , pred_boxes ):
#     xA = K.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
#     yA = K.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
#     xB = K.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
#     yB = K.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
#     interArea = K.maximum( 0.0 , xB - xA ) * K.maximum( 0.0 , yB - yA )
#     boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
#     boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
#     iou = interArea / ( boxAArea + boxBArea - interArea )
#     return iou


def iou_metric(y_true, y_pred):
    return intersection_over_union(y_true, y_pred)


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = mean_iou(y_true, y_pred)
    custom_loss.__name__ = 'custom_loss'
    return mse + (1 - iou)


def iou(y_true, y_pred, label: int):
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())

    intersection = K.sum(y_true * y_pred)

    union = K.sum(y_true) + K.sum(y_pred) - intersection

    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    num_labels = K.int_shape(y_pred)[-1]
    total_iou = K.variable(0)

    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    mean_iou.__name__ = 'mean_iou'
    return total_iou / num_labels
