"""
This code computes overlap (intersection over union) of two circles, counts number of true positives, false positives
and false negatives as well as it computes recall, precision, F1-score and area-under-curve (AUC). It also plots success
rate at different overlap threshold ranging from 0 to 1. From this overlap success plot, we computed the AUC. The
detection quality is measured using mainly AUC and also using F1-score (F-score or F-measure).
"""

import numpy as np
import matplotlib.pyplot as plt


# Computes overlap (intersection over union) of two circles
def circle_intersection_over_union(circle1, circle2):
    # The format of the circles is (either detection or ground truth) is [xc, yc, r]
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    d = np.sqrt(d2)
    t = ((r1 + r2) ** 2 - d2) * (d2 - (r2 - r1) ** 2)
    if t > 0:  # The circles overlap
        intersection_area = r1 ** 2 * np.arccos((r1 ** 2 - r2 ** 2 + d2) / (2 * d * r1)) + \
                        r2 ** 2 * np.arccos((r2 ** 2 - r1 ** 2 + d2) / (2 * d * r2)) - 1 / 2 * np.sqrt(t)
    elif d > r1 + r2:  # The circles are disjoint
        intersection_area = 0
    else:  # One circle is contained entirely within the other
        intersection_area = np.pi * min(r1, r2) ** 2

    circle1_area = np.pi*r1**2
    circle2_area = np.pi*r2**2

    # Divide the intersection by union of bboxA and bboxB: bboxA U bboxB = bboxA +  bboxB - intersectAB
    overlap_ratio = intersection_area / float(circle1_area + circle2_area - intersection_area)

    # Return the intersection over union value
    return overlap_ratio


# Example
circle1 = [10, 10, 6]
circle2 = [8, 8, 8]

iou = circle_intersection_over_union(circle1, circle2)
print('Example: ', iou)

# ========== Compute Recall, Precision, F1_score  and AUC =============================================================

# This example is just for two frames ------ Can be extended for as many frames as we want!
# Here zero-padding may be necessary to make equal matrix dimensions
groundtruth1 = [[8, 8, 6], [16, 16, 8], [30, 30, 10]]  # For frame 1
detections1 = [[10, 10, 6], [20, 20, 8], [5, 5, 4], [40, 30, 12]]

groundtruth2 = [[8, 8, 6], [16, 16, 8], [30, 30, 10]]  # For frame 2
detections2 = [[10, 10, 6], [20, 20, 8], [16, 16, 10], [40, 30, 12]]

N_frames = 2
dim1_det = max(np.array(detections1).shape[0], np.array(detections2).shape[0])
dim2_det = max(np.array(detections1).shape[1], np.array(detections2).shape[1])

dim1_gt = max(np.array(groundtruth1).shape[0], np.array(groundtruth2).shape[0])
dim2_gt = max(np.array(groundtruth1).shape[1], np.array(groundtruth2).shape[1])

groundtruth_total = np.zeros((N_frames, dim1_gt, dim2_gt)) # combine all groundtruths of all frames, in this case 2 frames
groundtruth_total[0] = groundtruth1
groundtruth_total[1] = groundtruth2

detections_total = np.zeros((N_frames, dim1_det, dim2_det)) # combine all detections of all frames, in this case 2 frames
detections_total[0] = detections1
detections_total[1] = detections2

detections_iou_total = []
groundTruth_iou_total = []
for d in range(groundtruth_total.shape[0]):  # for each frame d
    detections_d = detections_total[d]
    groundtruth_d = groundtruth_total[d]
    detection_groundtruth_iou = np.zeros((detections_d.shape[0], groundtruth_d.shape[0]))
    for gt in range(len(groundtruth_d)):
        for dt in range(len(detections_d)):
            det_circle = detections_d[dt]
            gt_circle = groundtruth_d[gt]
            iou = circle_intersection_over_union(det_circle, gt_circle)
            detection_groundtruth_iou[dt][gt] = iou
    print('detection_groundtruth_iou: ')
    print(detection_groundtruth_iou)
    detections_iou = list(np.max(detection_groundtruth_iou, 1))
    groundTruth_iou = list(np.max(detection_groundtruth_iou, 0))
    print('detections_iou:', detections_iou)
    print('groundtruth_iou:', groundTruth_iou)
    detections_iou_total = detections_iou_total + detections_iou
    groundTruth_iou_total = groundTruth_iou_total + groundTruth_iou

print('detections_iou_total: ', detections_iou_total)
print('groundTruth_iou_total: ', groundTruth_iou_total)


TP = [i for i in detections_iou_total if i >= 0.5]  # True positive
TP_num = len(TP)
FP = [i for i in detections_iou_total if i < 0.5]  # False positive
FP_num = len(FP)
FN = [i for i in groundTruth_iou_total if i < 0.5]  # False negative (miss-detections)
FN_num = len(FN)

Recall = TP_num/float(TP_num + FN_num)  # Recall or Sensitivity
Precision = TP_num/float(TP_num + FP_num)  # Precision
F1_score = 2 * Precision*Recall / (Precision + Recall)  # ..............This is a good measure.

print('TP_num:', TP_num)
print('FP_num:', FP_num)
print('FN_num:', FN_num)
print('Recall:', Recall)
print('Precision:', Precision)
print('F1_score:', F1_score)


# Plot Success rate
def frange(start, end, step):
    tmp = start
    while tmp < end:
        yield tmp
        tmp += step


threshold_list = []
Success_num_list = []
for i in frange(0, 1, 0.001):
    threshold_i = i
    Success = [i for i in detections_iou if i >= threshold_i]  # True positive
    Success_num = len(Success)
    threshold_list.append(threshold_i)
    Success_num_list.append(Success_num)

Success_max = max(Success_num_list)
Success_rate = [x / Success_max for x in Success_num_list]
print('threshold_list: ', threshold_list)
print('Success_rate: ', Success_rate)

# Compute Area Under Curve (AUC)
auc = np.trapz(Success_rate, threshold_list, 0.001, axis=0)  # This is the most quantitative measure to be used!
print('Area under curve; ', auc)

plt.figure(figsize=(40, 40))
plt.plot(threshold_list, Success_rate)  # For visual viewing, from which AUC is computed
plt.ylabel('Success rate')
plt.xlabel('Overlapp threshold')
plt.legend(['Overlap success plot'])
plt.show()
