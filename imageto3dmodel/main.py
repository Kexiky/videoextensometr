import cv2
import matplotlib.pyplot as plt

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

    pictureimshow([img_l, img_r], (1, 2), ['left', 'right'])

