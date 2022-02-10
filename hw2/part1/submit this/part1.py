import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mpyeditor
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Part 1

vid = mpy.VideoFileClip('shapes_video.mp4')

frame_count = vid.reader.nframes # 92 frames
video_fps = vid.fps

print('frame_count: ', frame_count)
print('video_fps: ', video_fps)


# height width channels = image.shape
def median_filter(image_rgb):
    '''
        This median filter processes RGB channels
        using zero padding with appropriate filter mask size,
        after noise removal it removes zero paddings.
    '''
    fw, fh = 3, 3
    window_size = fw * fh
    mid_index = (window_size) // 2

    resulting_image = image_rgb.copy()
    zero_padded_image = np.zeros(shape=(frame.shape[0] + (2 * (fh//2)), frame.shape[1] + (2 * (fw//2)), frame.shape[2]))
    zero_padded_image[fh//2:zero_padded_image.shape[0] - 2 * (fh//2) + 1,
                      fw//2:zero_padded_image.shape[1] - 2 * (fw//2) + 1, :] = image_rgb
    for channel, image in enumerate(cv2.split(zero_padded_image)):
        h, w = image.shape[:2]
        image_one_channel = np.zeros_like(image)
        image_window = np.zeros(shape=(window_size))
        half_fw = fw // 2
        half_fh = fh // 2
        for x in range(half_fw, w - half_fw):
            for y in range(half_fh, h - half_fh):
                window_index = 0
                for fx in range(0, fw):
                    for fy in range(0, fh):
                        y_index = y + fy - half_fh
                        x_index = x + fx - half_fw
                        image_window[window_index] = image[y_index][x_index]
                        window_index += 1
                image_window.sort()
                pixel_value = image_window[mid_index]
                if pixel_value > 255:
                    image_one_channel[y][x] = 255
                elif pixel_value < 0:
                    image_one_channel[y][x] = 0
                else:
                    image_one_channel[y][x] = pixel_value
                
        resulting_image[:, :, channel] = image_one_channel[fh//2:zero_padded_image.shape[0] - 2 * (fh//2) + 1,fw//2:zero_padded_image.shape[1] - 2 * (fw//2) + 1]
    
    return resulting_image
    

images_list = []
for i in range(frame_count):
    # type(frame): <class 'numpy.ndarray'>
    frame = vid.get_frame(i * 1.0 / video_fps)
    median_applied_frame = median_filter(frame)
    images_list.append(median_applied_frame)

clip = mpyeditor.ImageSequenceClip(images_list , fps = 25)
clip.write_videofile('part1_video.mp4' , codec='libx264')

