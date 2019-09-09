import cv2
import sys
import glob
import os
def video2frame(filename, split_ratio=0.9):
    vidcap = cv2.VideoCapture(filename + '.mp4')
    image_folder = filename + '/frames/'
    counter = 0
    while(True):
        if counter % 100 == 0:
            print(counter)
        success,image = vidcap.read()
        if success == True:
            cv2.imwrite(image_folder + str(counter).zfill(5) + '.png', image)
            counter += 1
        else:
            break
def crop_images(filename):
    image_folder = filename + '/frames/'
    frames = glob.glob(os.path.join(image_folder, '*.png'))
    counter = 0
    for name in frames:
        if counter % 100 == 0:
            print(counter)
        img = cv2.imread(name)
        crop_img = img[180:350, 70:570]
        #print(name.split('/')[3:])
        tmp = name.split('/')
        #print('/'.join(tmp[:2]) + '/frames_c/' + tmp[3])
        cv2.imwrite('/'.join(tmp[:2]) + '/frames_c/' + tmp[3], crop_img)
        #print(len(img), len(img[0]), len(img[0][0]))
        counter += 1

if __name__ == '__main__':
    video2frame(sys.argv[1])
    crop_images(sys.argv[1])
