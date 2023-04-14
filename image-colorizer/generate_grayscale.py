import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    directory = "data/thumbs/train"
    target_directory = "data/thumbs_gray/train"
    for count, filename in enumerate(os.listdir(directory)):
        im = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(target_directory, filename), img=im)
        print(f"Processed image {count + 1}")

    directory = "data/thumbs/test"
    target_directory = "data/thumbs_gray/test"
    for count, filename in enumerate(os.listdir(directory)):
        im = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(target_directory, filename), img=im)
        print(f"Processed image {count + 1}")

if __name__ == "__main__":
    main()