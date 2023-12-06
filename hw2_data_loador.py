import torchvision
import torch

class_to_idx = {
    "car": 0,
    "bird": 1,
    "bus": 2,
    "cat": 3,
    "dog": 4,
    "aeroplane": 5,
    "person": 6,
    "bicycle": 7,
    "motorbike": 8,
    "tvmonitor": 9,
    "bottle": 10,
    "sofa": 11,
    "diningtable": 12,
    "horse": 13,
    "train": 14,
    "pottedplant": 15,
    "sheep": 16,
    "chair": 17,
    "boat": 18,
    "cow": 19,
}


def collate_fn(batch):
    return tuple(zip(*batch))


def read_datasets(
    root_folder, image_set="train", transform=None, batches=5
) -> torch.utils.data.DataLoader:
    data = torchvision.datasets.VOCDetection(
        root_folder,
        year="2008",
        image_set=image_set,
        download=False,
        transform=transform,
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batches,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return data_loader


def convert_targets(objects, device):
    boxes = []
    labels = []
    for obj in objects:
        # Convert coordinates to float
        xmin, ymin, xmax, ymax = map(
            float,
            [
                obj["bndbox"]["xmin"],
                obj["bndbox"]["ymin"],
                obj["bndbox"]["xmax"],
                obj["bndbox"]["ymax"],
            ],
        )

        # Append to the boxes list
        box_np = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        label = obj["name"]  # You might want to map the label to an integer class ID
        label_id = class_to_idx[label]
        # if label_id > 4:
        #     continue
        boxes.append(box_np.to(device))
        # Append to the labels list (assuming you have a label mapping)

        labels.append(label_id)
    boxes = torch.stack(boxes, dim=0).to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)
    target = {
        "boxes": boxes,
        "labels": labels,
        "image_id": None,
    }
    return target
