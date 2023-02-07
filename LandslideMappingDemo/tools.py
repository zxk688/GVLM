import numpy as np
from PIL import Image


def normalize(img):
    min = img.min()
    max = img.max()
    x = (img - min) / (max - min) 
    return x


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def ComposePatches(results, X_points, Y_points,split_width):
    index = 0
    totalimgs = []
    maxh = len(Y_points)
    maxw = len(X_points)

    for i in range(0, maxh , 1):

        totalimg = results[index][:,0:X_points[1]-X_points[0]]
        index = index + 1
        for j in range(1, maxw-1, 1):

            totalimg = np.hstack((totalimg,results[index][:,0:(X_points[j+1]-X_points[j])]))

            index = index + 1
        totalimg = np.hstack((totalimg, results[index]))
        index = index + 1
        totalimgs.append(totalimg)

    lin = totalimgs[0]


    lin = lin[0:(Y_points[1] - Y_points[0]):, :]
    for i in range(1, maxh-1, 1):

        tt = totalimgs[i][0:(Y_points[i+1] - Y_points[i]), :]
        lin = np.vstack((lin,tt ))

    lin = np.vstack((lin, totalimgs[maxh-1]))

    return lin





