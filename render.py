import numpy as np

# matplotlib.pyplot.imshow(dog)
# matplotlib.pyplot.show()

from matplotlib import pyplot as plt
import cv2
import glob
import os
from PIL import Image


class render():
    def __init__(self, image_path="images/", video_path="videos/"):
        self.image_path = image_path
        self.video_path = video_path
        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
        if not os.path.exists(self.video_path):
            os.mkdir(self.video_path)
        self.num_images = 0
        self.frames_list = []

    # -------------------------------
    def add_frame_to_list(self, frame):
        self.frames_list.append(frame)

    # -------------------------------
    def generate_images_from_list(self, epoch):
        for frame in self.frames_list:
            self.generate_images(frame, epoch=epoch)
        self.num_images = 0
    # -------------------------------
    def reset_frame_list(self):
        self.frames_list = []

    # ------------------------------
    def generate_images(self, data, epoch):
        self.num_images += 1
        if not os.path.exists(f"{self.image_path}{epoch}"):
            os.makedirs(f"{self.image_path}{epoch}")

        img = Image.fromarray(data)
        img = img.convert("L")
        img.save(f"{self.image_path}{epoch}/{self.num_images}.jpg")

    # ----------------------------------------------------
    def generate_video_from_images(self, epoch):
       # print(f"Generating video at path: {self.video_path}{epoch}")
        if not os.path.exists(f"{self.video_path}{epoch}"):
            os.makedirs(f"{self.video_path}{epoch}")
        img_array = []
        for filename in glob.glob(f"{self.image_path}{epoch}/*.jpg"):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(f'{self.video_path}{epoch}{epoch}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
       # print(f"Finished generating video number: {epoch}")
    # ---------------------------------------------------
    def generate_video(self, epoch):
        self.generate_images_from_list(epoch=epoch)
        self.generate_video_from_images(epoch)
