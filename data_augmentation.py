import cv2
import numpy as np
import os

# Function to rotate the image
def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Function to flip (mirror) the image horizontally
def flip_image(image):
    return cv2.flip(image, 1)

# Function to adjust brightness
def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image

def augmentation(root_path):
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        cnt = 0
        for file in os.listdir(folder_path):
            rotate_random = np.arange(-30, 31, 5)
            np.random.shuffle(rotate_random)
            rotate_random = rotate_random[:4]
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)
            image_rotate1 = rotate_image(image, rotate_random[0])
            image_rotate2 = rotate_image(image, rotate_random[1])
            image_rotate3 = rotate_image(image, rotate_random[2])
            image_rotate4 = rotate_image(image, rotate_random[3])
            image_flip = flip_image(image)
            image_flip1 = flip_image(image_rotate1)
            image_flip2 = flip_image(image_rotate2)
            image_flip3 = flip_image(image_rotate3)
            image_flip4 = flip_image(image_rotate4)
            decrease_brightness = adjust_brightness(image, 0.8)
            increase_brightness = adjust_brightness(image, 1.2)
            ### save image
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_rotate1)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_rotate2)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_rotate3)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_rotate4)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_flip)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_flip1)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_flip2)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_flip3)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), image_flip4)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), decrease_brightness)
            cnt += 1
            cv2.imwrite(os.path.join(folder_path , f'{folder}{cnt}.png'), increase_brightness)
            cnt += 1

def main():
    augmentation('./Data/train')
    augmentation('./Data/test')

if __name__ == '__main__':
    main()