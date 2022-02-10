import numpy as np
import cv2
import moviepy.editor as moviepy

# extract headphones cat
cat = cv2.imread('cat-headphones.png')
background_height = cat.shape[0]
background_width = cat.shape[1]
ratio = 322 / background_height
new_width = int(background_width * ratio)
new_height = 322
cat = cv2.resize(src=cat, dsize=(new_width, new_height))
foreground = np.logical_or(cat[:, :, 1] < 180, cat[:, :, 0] > 150)
nonzero_x_cat, nonzero_y_cat = np.nonzero(foreground)
nonzero_cat_values = cat[nonzero_x_cat, nonzero_y_cat, :]

def find_projective_matrix(src_points, dst_points):
	matrix = []
	for sp, dp in zip(src_points, dst_points):
		matrix.append([sp[0], sp[1], 1, 0, 0, 0, -sp[0] * dp[0], -sp[1] * dp[0]])
		matrix.append([0, 0, 0, sp[0], sp[1], 1, -sp[0] * dp[1], -sp[1] * dp[1]])
	matrix = np.array(matrix, dtype=np.float32)
	N = matrix.reshape(8, 8)
	K = np.array(dst_points).reshape(8)
	intr_result = np.dot(np.dot(np.linalg.inv(np.dot(N.T, N)), N.T), K)
	proj_mat = np.append(np.array(intr_result).reshape(8), np.array(1, dtype = np.float32))
	proj_mat.resize((3, 3))
	return proj_mat

def find_new_points(pts):
    new_points = pts.reshape((4, 2))
    new_points = np.asarray(new_points, dtype=np.float32)
    new_points[[0, 1]] = new_points[[1, 0]]
    return new_points

def polygon_area(pts):
    pts = pts.reshape(4, 2)
    new_points = np.zeros((4, 2), dtype=np.uint64)
    new_points[1, 0], new_points[1, 1] = abs(pts[1, 0] - pts[0, 0]), abs(pts[1, 1] - pts[0, 1])
    new_points[2, 0], new_points[2, 1] = abs(pts[1, 0] - pts[0, 0]), abs(pts[2, 1] - pts[0, 1])
    new_points[3, 0], new_points[3, 1] = 0, abs(pts[3, 1] - pts[0, 1])
    new_points[[0, 1]] = new_points[[1, 0]]
    new_points = np.float32([new_points[0], new_points[1], new_points[2], new_points[3]]) 
    return ((new_points[3, 0] - new_points[0, 0] + 1) + ((new_points[2, 0] - new_points[1, 0] + 1)) * ((new_points[1, 1] - new_points[0, 1] + 1)) / 2) 

def perspective(blank_image, pts, img, rows, cols): #  -> blank_image with album cover
	blank_image = cv2.polylines(blank_image, [pts], True, (0, 255, 255))
	cv2.fillPoly(blank_image, pts = [pts], color =(0,255,255))
	foreground = np.logical_and.reduce((blank_image[:, :, 0] == 0, blank_image[:, :, 1] == 255, blank_image[:, :, 2] == 255))
	nonzero_x, nonzero_y = np.nonzero(foreground)

	src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
	dst_points = find_new_points(pts)
	projective_matrix = find_projective_matrix(src_points, dst_points)
	img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))

	foreground = np.logical_or.reduce((img_output[:, :, 1] != 0, img_output[:, :, 2] != 0, img_output[:, :, 0] != 0))
	nonzero_x2, nonzero_y2 = np.nonzero(foreground)

	try:
		blank_image[nonzero_x[:len(nonzero_x2)], nonzero_y[:len(nonzero_y2)], :] = img_output[nonzero_x2, nonzero_y2, :]
	except:
		pass

	foreground = np.logical_and.reduce((blank_image[:, :, 0] == 0, blank_image[:, :, 1] == 255, blank_image[:, :, 2] == 255))
	nonzero_x, nonzero_y = np.nonzero(foreground)
	blank_image[nonzero_x, nonzero_y] = [0, 0, 0]

	return blank_image

def find_rotation_matrix(center, angle, scale):
    angle = np.radians(angle) # turn angle into radians
    alpha = np.cos(angle)*scale
    beta = np.sin(angle)*scale
    return np.array([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
    ])

planes = np.zeros((9,472,4,3))
I = cv2.imread('munch.jpg')
rows, cols = I.shape[:2]

# part4
rot_image = I.copy()
# in order to rotate the image clockwise
# we send -60 as paramater and get the result
# center.jpg
# 
# We rotate around top left corner which is 
# 0,0 point when we view the image as a coordinate
# system, then we get result top_left.jpg 
rot_m1 = find_rotation_matrix((rows//2, cols//2), -60, 1)
rot_m2 = find_rotation_matrix((0, 0), -60, 1)
dst1 = cv2.warpAffine(rot_image, rot_m1, dsize=(cols, rows))
dst2 = cv2.warpAffine(rot_image, rot_m2, dsize=(cols, rows))
cv2.imwrite('center.jpg', dst1)
cv2.imwrite('top_left.jpg', dst2)
# part4

for i in range(1, 10):
	with open("Plane_"+str(i)+".txt") as f:
		content = f.readlines()
		for line_id in range(len(content)):
			sel_line = content[line_id]
			sel_line = sel_line.replace(')\n', '').replace("(", '').split(")")

			for point_id in range(4):
				sel_point = sel_line[point_id].split(" ")

				planes[i-1,line_id,point_id,0] = float(sel_point[0])
				planes[i-1,line_id,point_id,1] = float(sel_point[1])
				planes[i-1,line_id,point_id,2] = float(sel_point[2])

images_list = []
for i in range(0, 472):
	blank_image = np.zeros((322,572,3), np.uint8)
	pts_list = []
	for j in range(0, 9):
		pts = planes[j,i,:,:].squeeze()[:,0:2].astype(np.int32)
		temp = np.copy(pts[3,:])
		pts[3, :] = pts[2,:]
		pts[2, :] = temp
		pts_list.append(pts)
	
	pts_list.sort(key = lambda x: polygon_area(x), reverse=True) # sort for render order
	i = 0
	for pts in pts_list:
		pts = pts.reshape((-1, 1, 2))
		if i < len(pts_list) // 2:
			blank_image[nonzero_x_cat, nonzero_y_cat+ 572//3 - 20, :] = nonzero_cat_values
		i += 1
		blank_image = perspective(blank_image, pts, I, rows, cols)
	blank_image = blank_image[:, :, [2, 1, 0]]
	
	foreground = np.logical_and.reduce((blank_image[:, :, 1] == 0, blank_image[:, :, 2] == 0, blank_image[:, :, 0] == 0))
	nonzero_x, nonzero_y = np.nonzero(foreground)
	blank_image[nonzero_x, nonzero_y, :] = [255, 255, 255]

	images_list.append(blank_image)


clip = moviepy.ImageSequenceClip(images_list, fps = 25)
clip.write_videofile("part1_vid.mp4", codec="libx264")
