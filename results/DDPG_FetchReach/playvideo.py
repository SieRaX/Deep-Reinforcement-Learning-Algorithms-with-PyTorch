import cv2
import numpy as np
import glob

for filename in glob.glob('*.npy'):
	img_array = np.load(filename)
	shot, height, width, layers = img_array.shape
	size = (width, height)
	break

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

# print("Enter the number of episode you want to start: ", end = "")
# start = int(input())
# print("Enter the number of episode you want to end: ", end = "")
# end = int(input())

# i = 1
a = glob.glob('*.npy')
a.sort()
a.sort( key=lambda f: int(''.join(filter(str.isdigit, f))) )
for filename in a:

	img_array = np.load(filename)
	for i in range(shot):
			out.write(img_array[i])
	# if start <= i and i <= end:
	# 	img_array = np.load(filename)
	# 	for i in range(shot):
	# 			out.write(img_array[i])
	# i = i+1
	

out.release()