import os
import torchvision


if __name__ == "__main__":
    data_path = "/home/zlz/cv_dl_course/Dataset/train/VOCdevkit/VOC2008/JPEGImages"
    outFile = open(
        "/home/zlz/cv_dl_course/Dataset/train/VOCdevkit/VOC2008/ImageSets/Main/train.txt",
        "w",
    )
    for file in os.listdir(data_path):
        if file.endswith(".jpg"):
            outFile.write(file[:-4] + "\n")

    torchvision.datasets.VOCDetection(
        "/home/zlz/cv_dl_course/Dataset/vali",
        year="2008",
        image_set="val",
        download=False,
    )
