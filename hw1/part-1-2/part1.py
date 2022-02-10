# %%
import numpy as np
import os
import cv2
import moviepy.editor as mpy
from numpy.core.fromnumeric import nonzero
import matplotlib.pyplot as plt

# %%

# helpers
def plot_histogram(hist, title, color):
    plt.plot(hist, color=color)
    plt.title("Avg Histogram of {}".format(title))
    plt.xlabel("pixel values")
    plt.ylabel("frequency")
    plt.show()

def plot_channels_data(data):
    for i, c in zip(data, ('b', 'g', 'r')):
        plt.plot(i, color=c)
        plt.xlabel("pixel values")
        plt.ylabel("frequency")
        plt.show()

# H
def get_channel_absolute_histograms(image):
    if len(image.shape) == 2: # (..., 3)
        b, g, r = [m.flatten() for m in np.split(image, 3, axis=1)]
    else:
        b, g, r = [m.flatten() for m in np.split(image, 3, axis=2)]
    hist_b, _ = np.histogram(b, bins=256, range=[0, 256])
    hist_g, _ = np.histogram(g, bins=256, range=[0, 256])
    hist_r, _ = np.histogram(r, bins=256, range=[0, 256])
    return (hist_b, hist_g, hist_r)

# h, p, pdf
def get_channel_relative_histograms(no_of_pixels, hists):
    A = no_of_pixels
    pdf_b = hists[0] / A
    pdf_g = hists[1] / A
    pdf_r = hists[2] / A
    return (pdf_b, pdf_g, pdf_r)

# CDF, c, P, PDF
def get_relative_cumulative_frequencies(pdfs):
    cdf_b = np.zeros_like(pdfs[0])
    cdf_g = np.zeros_like(pdfs[1])
    cdf_r = np.zeros_like(pdfs[2])
    for i in range(len(pdfs[0])):
        cdf_b[i] = np.sum(pdfs[0][:i + 1])
    for i in range(len(pdfs[1])):
        cdf_g[i] = np.sum(pdfs[1][:i + 1])
    for i in range(len(pdfs[2])):
        cdf_r[i] = np.sum(pdfs[2][:i + 1])
    return (cdf_b, cdf_g,cdf_r)

def get_pdfs(image_path):
    image = cv2.imread(image_path)
    image_histograms = get_channel_absolute_histograms(image)
    A = image.shape[0] * image.shape[1]
    # thb, thg, thr = image_histograms
    image_pdfs = get_channel_relative_histograms(A, image_histograms)
    return image_pdfs

def get_cdfs(image_path):
    image = cv2.imread(image_path)
    image_histograms = get_channel_absolute_histograms(image)
    A = image.shape[0] * image.shape[1]
    # thb, thg, thr = image_histograms
    image_pdfs = get_channel_relative_histograms(A, image_histograms)
    # tpdfb, tpdfg, tpdfr = image_pdfs
    image_cdfs = get_relative_cumulative_frequencies(image_pdfs)
    tcdfb, tcdfg, tcdfr = image_cdfs
    return tcdfb, tcdfg, tcdfr

def get_cdfs_image(image):
    image_histograms = get_channel_absolute_histograms(image)
    A = image.shape[0] * image.shape[1]
    image_pdfs = get_channel_relative_histograms(A, image_histograms)
    return get_relative_cumulative_frequencies(image_pdfs)
    
def generate_LUT_for_histogram_matching(scdfs, tcdfs): # implemented this for three channels
    __LUT = np.zeros((256, 3))
    for scdf, tcdf, i in zip(scdfs, tcdfs, range(3)):
        gj = 0
        for gi in range(256):
            while tcdf[gj] < scdf[gi] and gj < 255:
                gj += 1
            __LUT[gi, i] = gj
            gj = 0
    return __LUT

def histogram_matching(__LUT, image):
    K1 = __LUT[image[:, 0], 0]
    K2 = __LUT[image[:, 1], 1]
    K3 = __LUT[image[:, 2], 2]
    K = np.vstack((K1, K2, K3)).T
    return K

def histogram_matching_3d(__LUT, image):
    K1 = __LUT[image[:, :, 0], 0]
    K2 = __LUT[image[:, :, 1], 1]
    K3 = __LUT[image[:, :, 2], 2]
    K = np.vstack((K1, K2, K3)).T
    return K
# helpers


# %%
# background operations
background = cv2.imread('./Malibu.jpg')
background_height = background.shape[0]
background_width = background.shape[1]
# print("background.shape: ", background.shape) # (776, 1998, 3)
ratio = 360 / background_height
new_width = int(background_width * ratio)
new_height = 360
## paramater dsize takes (xsize, ysize) --- (width height)
## ndarray.shape returns (height, width)
background = cv2.resize(src=background, dsize=(new_width, new_height))
# print("new background.shape: ", background.shape) # (360, 926, 3)

# part2 start
avg_hist_b = np.zeros(256, dtype=np.float64)
avg_hist_g = np.zeros(256, dtype=np.float64)
avg_hist_r = np.zeros(256, dtype=np.float64)
avg_pdf_b = np.zeros(256, dtype=np.float64)
avg_pdf_g = np.zeros(256, dtype=np.float64)
avg_pdf_r = np.zeros(256, dtype=np.float64)

# target image
tcdfb, tcdfg, tcdfr = get_cdfs('target.jpg')
# part2 end

# cat operations
images_list = []
main_dir = '.'
for number in range(180):
    image = cv2.imread(main_dir + '/cat/cat_{}.png'.format(number))
    # print("cat image shape: ", image.shape) # (360, 640, 3)

    # The pixels having a cat image.
    foreground = np.logical_or(image[:, :, 1] < 180, image[:, :, 0] > 150)
    # print("fg shape: ", foreground.shape) # (360, 640)
    # The ’foreground ’ variable here is only a True−False 
    # map with the same size. 
    # Using np.nonzero function we can find the locations 
    # of True values.
    nonzero_x, nonzero_y = np.nonzero(foreground)
    # A matrix of shape (..., 3) containing the pixel 
    # values belonging to the cat part.
    nonzero_cat_values = image[nonzero_x, nonzero_y, :]


    # part2
    # numpy reads images as BGR
    hist_b, hist_g, hist_r = get_channel_absolute_histograms(nonzero_cat_values)
    pdf_b, pdf_g, pdf_r = get_channel_relative_histograms(len(nonzero_cat_values), (hist_b, hist_g, hist_r))
    avg_hist_b += (hist_b / 180)
    avg_hist_g += (hist_g / 180)
    avg_hist_r += (hist_r / 180)
    avg_pdf_b += (pdf_b / 180)
    avg_pdf_g += (pdf_g / 180)
    avg_pdf_r += (pdf_r / 180)

    # source cdfs
    source_image = nonzero_cat_values.copy()
    scdfb, scdfg, scdfr = get_relative_cumulative_frequencies((pdf_b, pdf_g, pdf_r)) 
    __LUT = generate_LUT_for_histogram_matching((scdfb, scdfg, scdfr), (tcdfb, tcdfg, tcdfr))
    K = histogram_matching(__LUT, source_image)
    # part2

    new_frame = background.copy()
    # The cat part is placed to the previously obtained indices.
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
    
    # symmetry
    flipped_y = new_frame.shape[1] - 1 - nonzero_y
    new_frame[nonzero_x, flipped_y, :] = K
    # symmetry
    
    # The frame here is currently in RGB order.
    # However, the moviepy library uses BGR order as default.
    # Thus, it may be good to reverse the channels.
    new_frame = new_frame[:, :, [2, 1, 0]]
    images_list.append(new_frame)

clip = mpy.ImageSequenceClip(images_list, fps=25)
with mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration) as audio:
    clip = clip.set_audio(audio)
    # changed default write_videofile code for quicktime macOS audio issue
    # https://github.com/Zulko/moviepy/issues/51
    clip.write_videofile('part2_video.mp4', 
                         codec='libx264', 
                         audio_codec='aac',
                         temp_audiofile='temp-audio.m4a',
                         remove_temp=True
                         )

# %% [markdown]
# # Part 2 - Average cat histogram

# %%
plt.figure(1)
plt.plot(avg_hist_b, color='b')
plt.title(label='Average hist of blue channel')
plt.xlabel('pixel intensity values')
plt.ylabel('probability')
plt.savefig('avg_hist_b.png', dpi=300, bbox_inches='tight')

plt.plot(avg_hist_g, color='g')
plt.title(label='Average hist of green channel')
plt.xlabel('pixel intensity values')
plt.ylabel('probability')
plt.savefig('avg_hist_g.png', dpi=300, bbox_inches='tight')

plt.plot(avg_hist_r, color='r')
plt.title(label='Average hist of red channel')
plt.xlabel('pixel intensity values')
plt.ylabel('probability')
plt.savefig('avg_hist_r.png', dpi=300, bbox_inches='tight')

plt.figure(2)
plt.plot(avg_pdf_b, color='b')
plt.title(label='Average pdf of blue channel')
plt.xlabel('pixel intensity values')
plt.ylabel('probability')
plt.savefig('avg_pdf_b.png', dpi=300, bbox_inches='tight')

plt.plot(avg_pdf_g, color='g')
plt.title(label='Average pdf of green channel')
plt.xlabel('pixel intensity values')
plt.ylabel('probability')
plt.savefig('avg_pdf_g.png', dpi=300, bbox_inches='tight')

plt.plot(avg_pdf_r, color='r')
plt.title(label='Average pdf of red channel')
plt.xlabel('pixel intensity values')
plt.ylabel('probability')
plt.savefig('avg_pdf_r.png', dpi=300, bbox_inches='tight')


# %% [markdown]
# 


