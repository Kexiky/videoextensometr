import cv2
import matplotlib.pyplot as plt
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def pltnet(gridsize):
    ax = []
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            ax.append(plt.subplot2grid(gridsize, (i, j)))
    return ax

def pictureimshow(x, gridsize=(1, 1), title=["Image"], cmap="viridis"):
    fig = plt.figure()
    ax = pltnet(gridsize)
    for i in range(len(x)):
        ax[i].imshow(x[i], cmap=cmap)
        ax[i].set_title(title[i])
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_img_L = r'Images\IMG_L.jpg'
    path_img_R = r'Images\IMG_R.jpg'

    img_l = cv2.imread(path_img_L)
    img_r = cv2.imread(path_img_R)

    minDisparity = 0
    numDisparities = 64
    blockSize = 8
    disp12MaxDiff = 1
    uniquenessRatio = 10
    speckleWindowSize = 10
    speckleRange = 8

    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                   numDisparities=numDisparities,
                                   blockSize=blockSize,
                                   disp12MaxDiff=disp12MaxDiff,
                                   uniquenessRatio=uniquenessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange
                                   )

    disp = stereo.compute(img_l, img_r).astype(np.float32)
    disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
    pictureimshow([img_l, img_r, disp], (1, 3), ['left', 'right', 'disparity'])