import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    for i in range(1, 25001):
        im = cv2.imread(f"data/thumbs25k/im{i}.jpg", cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"im{i}.jpg", img=im)
        print(f"Processed image {i}")

if __name__ == "__main__":
    main()