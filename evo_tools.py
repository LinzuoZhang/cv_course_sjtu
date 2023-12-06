import numpy as np


def non_max_suppression(boxes, scores, iou_threshold):
    index_record = []
    selected_indices = np.argsort(scores)[::-1]
    boxes = np.array(boxes)
    while len(selected_indices) > 0:
        current_index = selected_indices[0]
        index_record.append(current_index)

        current_box = boxes[current_index]
        iou = calculate_iou(current_box, boxes[selected_indices[1:]])

        # Filter out boxes with high IoU
        selected_indices = selected_indices[1:][iou < iou_threshold]

    return index_record


def calculate_ap(precision, recall):
    # 计算平均精度
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 计算PR曲线下的面积
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def calculate_iou(box, boxes):
    intersection = np.maximum(
        0.0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])
    ) * np.maximum(
        0.0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])
    )
    union = (
        (box[2] - box[0]) * (box[3] - box[1])
        + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        - intersection
    )
    iou = intersection / union
    return iou


def calculate_iou_s(box1, box2):
    # 计算两个边界框的交并比
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    iou = intersection / max(union, 1e-10)
    return iou


def calculate_map(ground_truth, predictions, iou_threshold=0.5):
    # 初始化
    average_precisions = []

    for class_label in range(20):
        # 获取指定类别的真实标签和预测结果
        true_positives = []
        false_positives = []
        scores = []

        for i in range(len(predictions)):
            prediction = predictions[i]
            gt_boxes = ground_truth[i][class_label]["boxes"]
            gt_labels = ground_truth[i][class_label]["labels"]
            pred_boxes = prediction[class_label]["boxes"]
            pred_scores = prediction[class_label]["scores"]
            gt_boxes = np.array(gt_boxes)
            pred_boxes = np.array(pred_boxes)
            # 标记真正例和假正例
            assigned_gt = []
            for j in range(len(gt_boxes)):
                assigned = False
                for k in range(len(pred_boxes)):
                    if k in assigned_gt:
                        continue

                    iou = calculate_iou_s(gt_boxes[j], pred_boxes[k])
                    if iou >= iou_threshold and gt_labels[j] == class_label:
                        true_positives.append(1)
                        false_positives.append(0)
                        scores.append(pred_scores[k])
                        assigned_gt.append(k)
                        assigned = True
                        break

                if not assigned and gt_labels[j] == class_label:
                    true_positives.append(0)
                    false_positives.append(1)
                    scores.append(0)

        # 计算精度和召回率
        num_annotations = sum([len(gt[class_label]["boxes"]) for gt in ground_truth])
        num_predictions = len(true_positives)
        if num_annotations == 0 or num_predictions == 0:
            average_precisions.append(0)
            continue

        sorted_indices = np.argsort(scores)[::-1]
        true_positives = np.array(true_positives)[sorted_indices]
        false_positives = np.array(false_positives)[sorted_indices]

        cum_true_positives = np.cumsum(true_positives)
        cum_false_positives = np.cumsum(false_positives)

        precision = cum_true_positives / (cum_true_positives + cum_false_positives)
        recall = cum_true_positives / num_annotations

        # 计算平均精度并添加到列表
        ap = calculate_ap(precision, recall)
        average_precisions.append(ap)

    # 计算mAP
    mAP = np.mean(average_precisions)
    return mAP, average_precisions
