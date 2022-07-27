from math import sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

''' Program by: Ethan S
Date: 7/21/2022
Notes: Image detection program that finds particles in an image and also has 2D and 3D graphs of pixel intensity. Left
clicking a mouse over a pixel in the image will get the signals of all pixels that have the same X or Y coordinates and
plot them on a 2D graph. Right clicking on a pixel in the image will display a 3D heatmap graph of the pixel intensity
of the 9x9 neighborhood of that pixel. I have commented on what lines of code/specific values can be edited for whatever
you want to do. Also, the right and left mouseclick functions can be combined together. Finally, the 3D graphing uses
mayavi instead of matplot.
'''


def find_coord(event, x, y, flags, param):  # mouse functions
    if event == cv2.EVENT_FLAG_LBUTTON:  # 2D Signal Graphs
        print('(', x, ',', y, ')')  # print pixel coordinate
        Pixel_XValue = []
        X = np.arange(0, 1272)  # X Dimensions of Image
        r, g, b = img[y, X, 2], img[y, X, 1], img[y, X, 0]
        numx_count = 0
        for num in X:
            mean = sqrt(0.241 * (r[numx_count] ** 2) + 0.691 * (g[numx_count] ** 2) + 0.068 * (b[numx_count] ** 2))
            Pixel_XValue.append(mean)
            numx_count += 1
        plt.scatter(X, Pixel_XValue)
        plt.gca().update(dict(title='X-Cord Signal', xlabel='X', ylabel='Signal', xlim=None, ylim=None))  # set labels
        plt.show()  # for 2D X Signal graph

        Pixel_YValue = []
        Y = np.arange(0, 849)  # Y Dimensions of Image
        r, g, b = img[Y, x, 2], img[Y, x, 1], img[Y, x, 0]
        numy_count = 0
        for num in Y:
            mean = sqrt(0.241 * (r[numy_count] ** 2) + 0.691 * (g[numy_count] ** 2) + 0.068 * (b[numy_count] ** 2))
            Pixel_YValue.append(mean)
            numy_count += 1
        plt.scatter(Y, Pixel_YValue)
        plt.gca().update(dict(title='Y-Cord Signal', xlabel='Y', ylabel='Signal', xlim=None, ylim=None))  # set labels
        plt.show()  # for 2D Y Signal graph

    elif event == cv2.EVENT_FLAG_RBUTTON:  # find pixel intensity in 9x9 kernel w/ 3D graph
        print('(', x, ',', y, ')')  # print pixel coordinate
        Y = ([y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4], [y + 4],  # kernels
             [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3], [y + 3],
             [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2], [y + 2],
             [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1], [y + 1],
             [y], [y], [y], [y], [y], [y], [y], [y], [y],
             [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1], [y - 1],
             [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2], [y - 2],
             [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3], [y - 3],
             [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4], [y - 4])
        X = ([x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4],
             [x - 4], [x - 3], [x - 2], [x - 1], [x], [x + 1], [x + 2], [x + 3], [x + 4])
        r, g, b = img[Y, X, 2], img[Y, X, 1], img[Y, X, 0]  # get RGB values
        X, Y = np.array(X).flatten(), np.array(Y).flatten()  # convert set of lists into array
        num_count = 0
        Pixel_Intensity = []
        for num in Y:
            if num_count == 80:
                # print('(' + str(r[num_count])[1:-1] + ',' + str(g[num_count])[1:-1] + ',' + str(b[num_count])[1:-1] + ')')
                r_int, g_int, b_int = int((str(r[num_count])[1:-1])), int((str(g[num_count])[1:-1])), \
                                      int((str(b[num_count])[1:-1]))
                mean = sqrt(0.241 * (r_int ** 2) + 0.691 * (g_int ** 2) + 0.068 * (b_int ** 2))
                Pixel_Intensity.append(mean)
                # print('Pixel Intensity: ' + str(mean))
                # print('')
                Z = np.array(Pixel_Intensity)
                pts = mlab.points3d(X, Y, Z, Z)
                mesh = mlab.pipeline.delaunay2d(pts)
                pts.remove()
                surf = mlab.pipeline.surface(mesh)
                mlab.xlabel("X")
                mlab.ylabel("Y")
                mlab.zlabel("Signal")
                mlab.show()  # 3D heatmap for 9x9 kernel
                return
            if num_count < 80:
                # print('(' + str(r[num_count])[1:-1] + ',' + str(g[num_count])[1:-1] + ',' + str(b[num_count])[1:-1] + ')')
                r_int, g_int, b_int = int((str(r[num_count])[1:-1])), int((str(g[num_count])[1:-1])), \
                                      int((str(b[num_count])[1:-1]))
                mean = sqrt(0.241 * (r_int ** 2) + 0.691 * (g_int ** 2) + 0.068 * (b_int ** 2))
                Pixel_Intensity.append(mean)
                # print('Pixel Intensity: ' + str(mean))
                num_count += 1


if __name__ == "__main__":
    img = cv2.imread('C:/Users/admin/Downloads/DSC00068.JPG')  # set image file path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]  # create thresh image
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # define contours
    contours = contours[0] if len(contours) == 2 else contours[1]
    index = 1
    cluster_count = 0
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if 5000 > area > 10:  # change area value for threshold
            # draw contours
            cv2.drawContours(img, [cntr], 0, (0, 0, 255), 5)
            cluster_count = cluster_count + 1
            # the following is for getting area LISTED above the contour
            n, i = cntr.ravel(), 0
            for j in n:
                if i % 2 == 0:
                    x, y = n[i], n[i + 1]
                    if i == 0:
                        # change text size and color here
                        cv2.putText(img, str(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))
    index = index + 1

    print('number of particles detected:', cluster_count)
    p = .3  # resize image here, .3 = 30% of original size
    w, h = int(img.shape[1] * p), int(img.shape[0] * p)
    img = cv2.resize(img, (w, h))
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", find_coord)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
