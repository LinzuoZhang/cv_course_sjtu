import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import hw2_data_loador
import rcnn_model
import random

from PIL import Image, ImageDraw

import evo_tools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = rcnn_model.build_torchvision_model(device)
model.load_state_dict(torch.load("/home/zlz/cv_dl_course/weights/model.pth"))

model.eval()
model.to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

val_dataloader = hw2_data_loador.read_datasets(
    "/home/zlz/cv_dl_course/Dataset/vali", "val", transform, 1
)
random_indices = random.sample(range(len(val_dataloader)), 10)
id_to_class = {v: k for k, v in hw2_data_loador.class_to_idx.items()}
count = 0
with torch.no_grad():
    pbar = tqdm(val_dataloader)
    ground_truth = []
    predictions = []
    for i, (images, targets) in enumerate(pbar):
        ground_truth.append([{"boxes": [], "labels": []} for _ in range(20)])
        predictions.append(
            [{"boxes": [], "labels": [], "scores": []} for _ in range(20)]
        )
        images_i = list(image.to(device) for image in images)
        targets_processed = []
        for target in targets:
            target = hw2_data_loador.convert_targets(
                target["annotation"]["object"], device
            )
            targets_processed.append(target)
        outputs = model(images_i)

        res = [
            (
                target["boxes"].cpu().numpy().tolist(),
                target["labels"].cpu().numpy().tolist(),
                output["boxes"].cpu().numpy().tolist(),
                output["labels"].cpu().numpy().tolist(),
                output["scores"].cpu().numpy().tolist(),
            )
            for target, output in zip(targets_processed, outputs)
        ]

        for i in range(len(res)):
            target = res[i][0]
            target_label = res[i][1]
            output = res[i][2]
            output_label = res[i][3]
            score = res[i][4]
            output_indexes = evo_tools.non_max_suppression(
                boxes=output, scores=score, iou_threshold=0.3
            )
            nms_out = []
            nms_label = []
            nms_score = []
            for output_index in output_indexes:
                nms_out.append(output[output_index])
                nms_label.append(output_label[output_index])
                nms_score.append(score[output_index])
            for boxi, labeli in zip(target, target_label):
                ground_truth[i][labeli]["boxes"].append(boxi)
                ground_truth[i][labeli]["labels"].append(labeli)
            for boxi, labeli, scorei in zip(nms_out, nms_label, nms_score):
                predictions[i][labeli]["boxes"].append(boxi)
                predictions[i][labeli]["labels"].append(labeli)
                predictions[i][labeli]["scores"].append(scorei)

            image_path = (
                "/home/zlz/cv_dl_course/Dataset/vali/VOCdevkit/VOC2008/JPEGImages/"
                + targets[0]["annotation"]["filename"]
            )
            if count % 100 == 0:
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)

                box_count = 0
                for box in target:
                    draw.rectangle(box, outline="red")
                    draw.text((box[0], box[1]), id_to_class[target_label[box_count]])
                    box_count = box_count + 1
                box_count = 0
                for box in nms_out:
                    draw.rectangle(box, outline="green")
                    draw.text(
                        (box[0], box[1]),
                        f"{id_to_class[nms_label[box_count]]} {nms_score[box_count]: .2f}",
                    )
                    box_count = box_count + 1

                img.save(f"result/{count}.jpg")

            count += 1

    mAP, average_precisions = evo_tools.calculate_map(ground_truth, predictions)
    print(f"mAP: {mAP}")
    import matplotlib.pyplot as plt

    categories = []
    values = []
    for label in range(20):
        categories.append(id_to_class[label])
        values.append(average_precisions[label])
    plt.bar(categories, values)
    plt.xticks(rotation=45, ha="right")  # rotation为旋转角度，ha为对齐方式

    # 设置图表标题和轴标签
    plt.title("Bar Chart Example")
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.savefig("result/mAP.png")
