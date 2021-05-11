import tensorflow as tf


def interval_overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2

    if x3 < x1:
        return 0 if x4 < x1 else (min(x2, x4) - x1)
    else:
        return 0 if x2 < x3 else (min(x2, x4) - x3)


def intersection_over_union(box1, box2):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.ymax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect_area = intersect_h * intersect_w

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymax
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymax

    union_area = w1 * h1 + w2*h2 - intersect_area

    return float(intersect_area) / union_area


def iou_metric(y_true, y_pred):
    return intersection_over_union(y_true, y_pred)
