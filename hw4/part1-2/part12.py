'''
    Part 1.2
    Background Subtraction to detect moving objects

    If we know what the background looks like, it is easy to identify “interesting bits” Idea: use a temporal moving average to estimate background (or reference) image
    which has stationary components
    * Compare this frame with (i.e. subtract) subsequent frames including a moving object
    * Cancels out stationary elements
    * Large absolute values are “interesting pixels”


    We use Median Method for Background Subtraction:
    Idea: Relative to the consistent background, the moving objects are just temporal outliers, 
    so: Use median filter across time
'''
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mpyeditor

# read file names
DIR = './DJI_0101/'
file_names = sorted([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
# print(len(file_names)) # 459 images

# pick an image from every second # assuming 24 fps
selected_file_names = [file_name for file_name in file_names[::24]] # selected 20 frames
# selected_file_names = [file_name for file_name in file_names] # selected 20 frames
selected_images = [cv2.imread(os.path.join(DIR, file_name), 0) for file_name in selected_file_names]


temp_images = np.zeros(shape=selected_images[0].shape + (len(selected_images),))
for idx, image in enumerate(selected_images):
    temp_images[:,:,idx] = image

# background image
median_image = np.median(temp_images, axis=2).astype(np.uint8)
background_image = median_image.copy()


images_list = []
for file_name in file_names:
    image = cv2.imread(os.path.join(DIR, file_name), 0)
    # Background Removal
    motion_detected_image = cv2.absdiff(background_image, image)

    # thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(motion_detected_image, (5,5), 0)
    _, thresholded_image = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    thresholded_image = cv2.dilate(thresholded_image, None, iterations=2)
    thresholded_image = cv2.erode(thresholded_image, None, iterations=2)

    frame = thresholded_image
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    images_list.append(frame)
    # print("frame: ", frame)
    # break
    
    # images_list.append(image)

print(len(images_list))
print(len(images_list[0]))
# print(images_list[0])
# print(file_name)

clip = mpyeditor.ImageSequenceClip(images_list , fps = 20, with_mask=False)
clip.write_videofile('part12_video.mp4' , codec='libx264')
# plt.imshow(th3, cmap='gray')
# plt.pause(10000)
