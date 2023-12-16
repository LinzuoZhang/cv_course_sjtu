import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import hw2_dataloador
import rcnn_model
import random

from PIL import Image, ImageDraw

import evo_tools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = rcnn_model.build_torchvision_model(device)
model.load_state_dict(torch.load("exps/train3/run25/model_40.pth"))

model.eval()
model.to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

val_dataset = hw2_dataloador.HW2VOCDataset(
    "/home/zlz/cv_dl_course/Data", "val", transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=hw2_dataloador.collate_fn,
)
random_indices = random.sample(range(len(val_dataloader)), 10)
id_to_class = {v: k for k, v in hw2_dataloador.class_to_idx.items()}
count = 0
with torch.no_grad():
    pbar = tqdm(val_dataloader)
    ground_truth = []
    predictions = []
    for _, (images, targets) in enumerate(pbar):
        ground_truth.append([{"boxes": [], "labels": []} for _ in range(6)])
        predictions.append(
            [{"boxes": [], "labels": [], "scores": []} for _ in range(6)]
        )
        images_i = list(image.to(device) for image in images)
        targets_processed = []
        for target in targets:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)

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
                boxes=output, scores=score, iou_threshold=0.9
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

            image_path = targets[0]["filename"]
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

    mAP, average_precisions = evo_tools.calculate_map(ground_truth, predictions, 0.7)
    print(f"mAP: {mAP}")
    import matplotlib.pyplot as plt

    categories = []
    values = []
    for label in range(5):
        categories.append(id_to_class[label + 1])
        values.append(average_precisions[label])
    plt.bar(categories, values)
    plt.xticks(rotation=45, ha="right")  # rotation为旋转角度，ha为对齐方式

    # 设置图表标题和轴标签
    plt.title("mAP")
    plt.xlabel("Categories")
    plt.ylabel("mAP")
    plt.savefig("result/mAP.png")
