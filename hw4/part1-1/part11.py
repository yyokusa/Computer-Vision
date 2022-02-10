'''
    Part 1.1
    Motion Estimation: The Lucas-Kanade Method
'''
import os
import numpy as np
import cv2
import moviepy.editor as mpyeditor

# read file names
DIR = './DJI_0101/'
file_names = sorted([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

# pick an image from every second # assuming 24 fps
selected_file_names = [file_name for file_name in file_names[::24]] # selected 20 frames
selected_images = [cv2.imread(os.path.join(DIR, file_name), 0) for file_name in selected_file_names]


temp_images = np.zeros(shape=selected_images[0].shape + (len(selected_images),))
for idx, image in enumerate(selected_images):
    temp_images[:,:,idx] = image

# background image
median_image = np.median(temp_images, axis=2).astype(np.uint8)
background_image = median_image.copy()




scale = 1
delta = 0
ddepth = cv2.CV_16S

def derivative_x(arr, row, col, window_size):
    derivative_arr = np.zeros((window_size, window_size))
    
    grad_x = cv2.Sobel(arr, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.convertScaleAbs(grad_x)
    for r in range(window_size):
        for c in range(window_size):
            if col + c + 1 > grad_x.shape[1] - 1 or row + r + 1 > grad_x.shape[0] - 1:
                derivative_arr[r][c] = grad_x[grad_x.shape[0] - 1][grad_x.shape[1] - 1]
            else:
                derivative_arr[r][c] = grad_x[row + r][col + c]
    print("derivative_arr: ", derivative_arr)
    return derivative_arr

def derivative_y(arr, row, col, window_size):
    derivative_arr = np.zeros((window_size, window_size))
    grad_y = cv2.Sobel(arr, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.convertScaleAbs(grad_y)
    for r in range(window_size):
        for c in range(window_size):
            if col + c + 1 > grad_y.shape[1] - 1 or row + r + 1 > grad_y.shape[0] - 1:
                derivative_arr[r][c] = grad_y[grad_y.shape[0] - 1][grad_y.shape[1] - 1]
            else:
                derivative_arr[r][c] = grad_y[row + r][col + c]
    return derivative_arr

def derivative_time(arr_1, arr_2, row, col, window_size):
    derivative_arr = np.zeros((window_size, window_size))
    for r in range(window_size):
        for c in range(window_size):
            
            if col + c > arr_2.shape[1] - 1 or row + r > arr_2.shape[0] - 1:
                derivative_arr[r][c] = arr_2[arr_2.shape[0] - 1][arr_2.shape[1] - 1] - arr_1[arr_1.shape[0] - 1][arr_1.shape[1] - 1]
            else:
                derivative_arr[r][c] =  arr_2[row + r][col + c] - arr_1[row + r][col + c]
    derivative_arr = cv2.convertScaleAbs(derivative_arr)
    return derivative_arr


window_size = 5

final_images = []
for img_index in range(len(file_names) - 1):
    loaded_image_1 = cv2.imread(os.path.join(DIR, file_names[img_index]), 0)
    loaded_image_2 = cv2.imread(os.path.join(DIR, file_names[img_index + 1]), 0)
    

    motion_detected_image = cv2.absdiff(background_image, loaded_image_2)
    blur = cv2.GaussianBlur(motion_detected_image, (5,5), 0)
    _, thresholded_image = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    thresholded_image = cv2.dilate(thresholded_image, None, iterations=2)
    thresholded_image = cv2.erode(thresholded_image, None, iterations=2)
    frame = thresholded_image

    p0 = cv2.goodFeaturesToTrack(loaded_image_1, mask = frame, maxCorners = 200, qualityLevel = 0.2, 
                                            minDistance = 5, blockSize = 5)
    x_features = [int(i[0][0]) for i in p0]
    y_features = [int(i[0][1]) for i in p0]
    
    velocity = [None] * len(x_features)
    for idx in range(len(x_features)):
        if velocity[idx] is not None:
            x_features[idx] += int(round(velocity[idx][0][0]))
            y_features[idx] += int(round(velocity[idx][1][0]))

        derivative_x_arr = derivative_x(loaded_image_1, y_features[idx], x_features[idx], window_size)
        derivative_y_arr = derivative_y(loaded_image_1, y_features[idx], x_features[idx], window_size)
        derivative_t_arr = derivative_time(loaded_image_1, loaded_image_2, y_features[idx], x_features[idx], window_size)

        derivative_x_arr_transpose = np.transpose(np.array([derivative_x_arr.flatten()]))
        derivative_y_arr_transpose = np.transpose(np.array([derivative_y_arr.flatten()]))
        derivative_t_arr_transpose = np.transpose(np.array([derivative_t_arr.flatten()]))

        s = np.concatenate(np.array([derivative_x_arr_transpose, derivative_y_arr_transpose]), axis=1)
        s_t_s = np.matmul(np.transpose(s), s)

        w, v = np.linalg.eig(s_t_s)
        temp_matrix = np.matmul(np.linalg.pinv(s_t_s), np.transpose(s))
        velocity[idx] = np.matmul(temp_matrix, derivative_t_arr_transpose)

        line_scale_factor = 50
        line_x0 = x_features[idx] + window_size
        line_y0 = y_features[idx] + window_size

        loaded_image_1 = cv2.arrowedLine(loaded_image_1, (int(line_x0), int(line_y0)), 
            (int(line_x0 + velocity[idx][0][0] * line_scale_factor), int(line_y0 + velocity[idx][1][0] * line_scale_factor)),
                (0, 0, 255), 3, tipLength = 0.5)

    # cv2.imshow("hey", loaded_image_1)
    # cv2.waitKey(10000)
    loaded_image_1 = cv2.cvtColor(loaded_image_1, cv2.COLOR_GRAY2BGR)
    final_images.append(loaded_image_1)


clip = mpyeditor.ImageSequenceClip(final_images , fps = 20, with_mask=False)
clip.write_videofile('part11_video.mp4' , codec='libx264')
