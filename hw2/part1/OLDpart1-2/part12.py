from __future__ import print_function
import cv2
import numpy as np
import random

threshold = 10
# max_threshold = 90
max_threshold = 100
random.seed(random.randint(1, 10000))

def corner_detection(src, val, imageName):
    count = 0
    image_copy = np.copy(src)
    threshold = max(val, 1)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if minEigenValues[i,j] > matrixMinVal + ( matrixMaxVal - matrixMinVal ) * threshold/max_threshold:
                cv2.circle(image_copy, (j,i), 4, (random.randint(0,256), random.randint(0,256), random.randint(0,256)), cv2.FILLED)
                count += 1
    print("HEREEEEEEE")
    cv2.imwrite('myShiTomasi_{}.png'.format(imageName), image_copy)
    return count

for i in range(20, 24):
    src = cv2.imread('extracted{}.png'.format(i), 0)
    minEigenValues = cv2.cornerMinEigenVal(src, 3, 3)
    matrixMinVal, matrixMaxVal, _, _ = cv2.minMaxLoc(minEigenValues)
    print(corner_detection(src, threshold, i))

exit()

# 627
# HEREEEEEEE
# 699 # 5gen ->  699 - 627 = 72
# HEREEEEEEE
# 765 # kare -> 765 - 699 = 66
# HEREEEEEEE
# 847 # 5gen -> 847 - 765 = 82
# HEREEEEEEE
# 996 # yildiz -> 996 - 847 = 149
# HEREEEEEEE
# 1081 # kare -> 1081 - 996 = 85
# HEREEEEEEE
# 1235 # yildiz -> 1235 - 1081 = 154


# HEREEEEEEE
# 1316 # kare -> 1316 - 1235 = 85
# HEREEEEEEE
# 1505 # yildiz -> 1505 - 1316 = 154
# HEREEEEEEE
# 1502 # 5gen -> 1502 - 1505 = -3
# HEREEEEEEE
# 1641 # 5gen -> 1641 - 1502 = 139



























edges = cv.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
cv.imwrite('canny.png', edges)
print(edges.size)
print(edges.shape)




# height width channels = image.shape
def median_filter(image):
    '''
        This median filter processes RGB channels
        using zero padding with appropriate filter mask size,
        after noise removal it removes zero paddings.
    '''
    

    listOfGXY = []
    fw, fh = 3, 3
    # window_size = fw * fh
    # mid_index = (window_size) // 2

    # resulting_image = image.copy()
    zero_padded_image = np.zeros(shape=(frame.shape[0] + (2 * (fh//2)), frame.shape[1] + (2 * (fw//2))))
    zero_padded_image[fh//2:zero_padded_image.shape[0] - 2 * (fh//2) + 1,
                      fw//2:zero_padded_image.shape[1] - 2 * (fw//2) + 1] = image
    # for channel, image in enumerate(cv2.split(zero_padded_image)):
    h, w = image.shape[:2]
    # image_one_channel = np.zeros_like(image)
    # image_window = np.zeros(shape=(window_size))
    image = zero_padded_image
    half_fw = fw // 2
    half_fh = fh // 2
    for x in range(half_fw, w - half_fw):
        for y in range(half_fh, h - half_fh):
            window_index = 0
            GXY = np.zeros(shape=(2,2))
            for fx in range(0, fw):
                for fy in range(0, fh):
                    y_index = y + fy - half_fh
                    x_index = x + fx - half_fw
                    # image_window[window_index] = image[y_index][x_index]
                    Ix = (image[y_index][x_index + 1] - image[y_index][x_index - 1]) // 2
                    Iy = (image[y_index + 1][x_index] - image[y_index - 1][x_index]) // 2
                    window_index += 1
                    GXY += np.array([[Ix**2, Ix*Iy], [Ix*Iy, Iy**2]])
            if GXY.all():
                listOfGXY.append(GXY)
                # image_window.sort()
                # pixel_value = image_window[mid_index]
                # if pixel_value > 255:
                #     image_one_channel[y][x] = 255
                # elif pixel_value < 0:
                #     image_one_channel[y][x] = 0
                # else:
                #     image_one_channel[y][x] = pixel_value
                
        # resulting_image[:, :, channel] = image_one_channel[fh//2:zero_padded_image.shape[0] - 2 * (fh//2) + 1,fw//2:zero_padded_image.shape[1] - 2 * (fw//2) + 1]
    
    # return resulting_image
    return listOfGXY
frame = edges
mfedges = median_filter(edges)
print(mfedges)
print(len(mfedges))







# print(len(edges[0]]))
# print(edges[0]])



# exit()















# from __future__ import print_function
# import cv2 as cv
# import argparse
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='part1_video.mp4')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# if not capture.isOpened():
#     print('Unable to open: ' + args.input)
#     exit(0)
# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
    
#     fgMask = backSub.apply(frame)
    
    
#     cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
#     cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
#     cv.imwrite('Frame.png', frame)
#     cv.imwrite('FG Mask.png', fgMask)
#     break
#     keyboard = cv.waitKey(30)
#     if keyboard == 'q' or keyboard == 27:
#         break