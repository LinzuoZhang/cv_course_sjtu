import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import os
import time

class_to_idx = {
    "car": 1,
    "bird": 2,
    "bus": 3,
    "cat": 4,
    "dog": 5,
    "aeroplane": 6,
    "person": 7,
    "bicycle": 8,
    "motorbike": 9,
    "tvmonitor": 10,
    "bottle": 11,
    "sofa": 12,
    "diningtable": 13,
    "horse": 14,
    "train": 15,
    "pottedplant": 16,
    "sheep": 17,
    "chair": 18,
    "boat": 19,
    "cow": 20,
}


def collate_fn(batch):
    return tuple(zip(*batch))


class HW2VOCDataset(Dataset):
    def __init__(self, root_dir, type="train", transform=None):
        self.type = type
        self.root_dir = root_dir
        self.transform = transform
        self.jpeg_root = os.path.join(root_dir, "JPEGImages")
        self.annotation_root = os.path.join(root_dir, "Annotations")
        xml_files = []
        self.annotions = []
        self.jpeg_files = []
        for file in os.listdir(self.jpeg_root):
            if os.path.isdir(os.path.join(self.jpeg_root, file)) and file.endswith(
                type
            ):
                jpeg_file_base = os.path.join(self.jpeg_root, file)
                annotions_file_base = os.path.join(self.annotation_root, file)
                for file in os.listdir(jpeg_file_base):
                    self.jpeg_files.append(os.path.join(jpeg_file_base, file))
                    xml_files.append(
                        os.path.join(annotions_file_base, file.split(".")[0] + ".xml")
                    )
        for xml_file in xml_files:
            annotations = self.parse_xml(xml_file)
            annotations["filename"] = xml_file.replace(
                "Annotations", "JPEGImages"
            ).replace(".xml", ".jpg")
            self.annotions.append(annotations)

    def __len__(self):
        return len(self.annotions)

    def __getitem__(self, idx):
        image = cv2.imread(self.jpeg_files[idx], cv2.IMREAD_UNCHANGED)
        annotations = self.annotions[idx]

        if self.transform:
            # image, boxes = self.transform_image_and_bbox(
            #     image, annotations["boxes"], self.transform
            # )
            image = self.transform(image)
        # annotations["boxes"] = boxes

        return image, annotations

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Append to the boxes list
            box_np = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
            label = obj.find("name").text

            label_id = class_to_idx[label]
            if label_id > 5:
                continue
            boxes.append(box_np)
            # Append to the labels list (assuming you have a label mapping)

            labels.append(label_id)
        boxes = torch.stack(boxes, dim=0)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": None,
        }
        return target

    def transform_image_and_bbox(self, image, bbox, transform):
        # 应用图像变换
        transformed_image = transform(image)

        # 更新边界框坐标
        # 假设 bbox 格式为 [xmin, ymin, xmax, ymax]
        original_width, original_height = image.size
        scale_x = 224 / original_width
        scale_y = 224 / original_height
        bbox[:, 0] = bbox[:, 0] * scale_x
        bbox[:, 1] = bbox[:, 1] * scale_y
        bbox[:, 2] = bbox[:, 2] * scale_x
        bbox[:, 3] = bbox[:, 3] * scale_y
        return transformed_image, bbox


if __name__ == "__main__":
    data = HW2VOCDataset("/home/zlz/cv_dl_course/Data")
    res = data.__getitem__(1387)
    print(res)
