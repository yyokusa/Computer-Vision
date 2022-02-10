import moviepy.video.io.VideoFileClip as mpy
import cv2
import numpy as np

from __future__ import print_function
import cv2
import numpy as np
import random

# Shape Count
# [46, 3, 45]

def corner_detection(src, val, imageName):
    count = 0
    image_copy = np.copy(src)
    threshold = max(val, 1)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if minEigenValues[i,j] > matrixMinVal + ( matrixMaxVal - matrixMinVal ) * threshold/max_threshold:
                cv2.circle(image_copy, (j,i), 4, (random.randint(0,256), random.randint(0,256), random.randint(0,256)), cv2.FILLED)
                count += 1
    cv2.imwrite('myShiTomasi_{}.png'.format(imageName), image_copy)
    return count

vid = mpy.VideoFileClip('part1_video.mp4')

frame_count = vid.reader.nframes # 92 frames
video_fps = vid.fps

print('frame_count: ', frame_count)
print('video_fps: ', video_fps)
shape_counts = [0, 0, 0] # 4gen 5gen yildiz
threshold = 10
max_threshold = 100
random.seed(random.randint(1, 10000))
prev_dot_count = 0

# prev_frame = vid.get_frame(0 * 1.0 / video_fps)
# prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# prev_frame = cv2.Canny(prev_frame,100,200)

for i in range(frame_count):
    frame = vid.get_frame(i * 1.0 / video_fps)
    frame = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    src = cv2.Canny(img,100,200)

    # temp = src
    # src = src - prev_frame
    # prev_frame = temp

    minEigenValues = cv2.cornerMinEigenVal(src, 3, 3)
    matrixMinVal, matrixMaxVal, _, _ = cv2.minMaxLoc(minEigenValues)
    dot_count = corner_detection(src, threshold, i)
    count = dot_count - prev_dot_count
    print(count)
    if count < 70:
        shape_counts[0] += 1
    elif count >= 70  and count <= 100:
        shape_counts[1] += 1
    elif count > 100:
        shape_counts[2] += 1
    prev_dot_count = dot_count
    

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


print(shape_counts)
