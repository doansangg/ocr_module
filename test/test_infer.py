import cv2 
import os 
import argparse
from reader import Reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='data/test/test.jpg')
    args = parser.parse_args()
    ocr_model = Reader()
    image = cv2.imread(args.image_path)
    output = ocr_model(image)
    print(output)