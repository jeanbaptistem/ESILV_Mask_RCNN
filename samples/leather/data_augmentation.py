import random
import os
import cv2
import numpy as np
from PIL import Image


def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def zoom(img, mask, file_name, mask_file_name):
    value = 0.3
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    im = Image.fromarray(img)
    im.save(file_name + "_z.png")
    if mask_file_name != None:
        mask = mask[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
        mask = fill(mask, h, w)
        im = Image.fromarray(mask)
        im.save(mask_file_name + "_z_mask.png")



def channel_shift(img, mask, file_name, mask_file_name):
    value = 0.5
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save(file_name + "_cs.png")
    if mask_file_name != None:
        mask = mask + value
        mask[:,:,:][mask[:,:,:]>255]  = 255
        mask[:,:,:][mask[:,:,:]<0]  = 0
        mask = mask.astype(np.uint8)
        im = Image.fromarray(mask)
        im.save(mask_file_name + "_cs_mask.png")



def horizontal_flip(img, mask, file_name, mask_file_name):
    img = cv2.flip(img, 1)
    im = Image.fromarray(img)
    im.save(file_name + "_hf.png")
    if mask_file_name != None:
        mask = cv2.flip(mask, 1)
        im = Image.fromarray(mask)
        im.save(mask_file_name + "_hf_mask.png")


def vertical_flip(img, mask, file_name, mask_file_name):
    img = cv2.flip(img, 0)
    im = Image.fromarray(img)
    im.save(file_name + "_vf.png")
    if mask_file_name != None:
        mask = cv2.flip(mask, 0)
        im = Image.fromarray(mask)
        im.save(mask_file_name + "_vf_mask.png")


def rotation(img, mask, file_name, mask_file_name):
    h, w = img.shape[:2]
    for _angle in [30, 60, 120, 150]:
        angle = int(random.uniform(-_angle, _angle))
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        im = Image.fromarray(img)
        im.save(file_name + "_{}r.png".format(_angle))
        if mask_file_name != None:
            mask = cv2.warpAffine(mask, M, (w, h))
            im = Image.fromarray(mask)
            im.save(mask_file_name + "_{}r_mask.png".format(_angle))


def blur_im(img, mask, file_name, mask_file_name):
    img = cv2.blur(img,(5,5))
    im = Image.fromarray(img)
    im.save(file_name + "_bl.png")
    if mask_file_name != None:
        mask = cv2.blur(mask,(5,5))
        im = Image.fromarray(mask)
        im.save(mask_file_name + "_bl_mask.png")


def brightness(img, mask, file_name, mask_file_name):
    low = 0.5
    high = 3
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    im = Image.fromarray(img)
    im.save(file_name + "_br.png")
    if mask_file_name != None:
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        im = Image.fromarray(mask)
        im.save(mask_file_name + "_br_mask.png")



if __name__ == "__main__":
    # Root directory of the project
    ROOT_DIR = os.path.abspath(".")
    LEATHER_DIR = os.path.join(ROOT_DIR, "datasets\leather")

    print("Dataset dir: ", LEATHER_DIR)
    subsets = ["train", "val"]
    for subset in subsets:
        dataset_dir_sub = os.path.join(LEATHER_DIR, subset)

        folders = ["color", "cut", "fold", "glue", "poke"]
        for label in folders:
            label_dir = os.path.join(dataset_dir_sub, label)
            image_ids = next(os.walk(label_dir))[2]

            for img in image_ids:
                path = os.path.join(dataset_dir_sub, label, img)
                image = cv2.imread(path)
                file_name = os.path.join(dataset_dir_sub, label, img.split('.')[0])
                
                mask = None
                mask_file_name = None
                if label != "good":
                    mask_path = os.path.join(LEATHER_DIR, "ground_truth", label, "{}_mask.png".format(img.split('.')[0]))
                    mask = cv2.imread(mask_path)
                    mask_file_name = os.path.join(LEATHER_DIR, "ground_truth", label, img.split('.')[0])

                print("Working on", file_name + ".png")
                zoom(image, mask, file_name, mask_file_name)
                brightness(image, mask, file_name, mask_file_name)
                channel_shift(image, mask, file_name, mask_file_name)
                horizontal_flip(image, mask, file_name, mask_file_name)
                vertical_flip(image, mask, file_name, mask_file_name)
                rotation(image, mask, file_name, mask_file_name)
                blur_im(image, mask, file_name, mask_file_name)



