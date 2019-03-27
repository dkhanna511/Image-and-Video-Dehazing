import cv2
import math
import numpy as np
import sys

def apply_mask(matrix, mask, fill_value):
    #print("MATRIX=", matrix)
    #print("mask=\n" ,mask)
    #print("fill value=\n", fill_value)
                 
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    print('MASKED=',masked)
    return masked.filled()

def apply_threshold(matrix, low_value=255, high_value=255):
    low_mask = matrix < low_value
    print("low mask=",low_mask)
    
    matrix = apply_mask(matrix, low_mask, low_value)
    print('Low MASK->',low_mask,'\nMatrix->',matrix)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100
    print("shape of image = ", img.shape[2])

    half_percent = percent / 200.0
    print('HALF PERCENT->',half_percent)

    channels = cv2.split(img)
    print('Channels->\n',channels)
    print('Shape->',channels[0].shape)
    print('Shape of channels->',len(channels[2]))

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2

	# find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        print('vec=',vec_size,'\nFlat=',flat)
        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]
        print("Number of columns = ", n_cols)

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        print("Lowval: ", low_val)
        print("Highval: ", high_val)
        print(flat[60])
        print(flat[11940])
        

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

if __name__ == '__main__':
    #img = cv2.imread(sys.argv[1])
	cap=cv2.VideoCapture('/media/dheeraj/9A26F0CB26F0AA01/WORK/github_repo/Dehazing/haze-videos/Whale.mov')
   	#img = cv2.imread('/home/dheeraj/Downloads/Whale.mov')
	while True:
	
		ret, frame=cap.read()
	
		out = simplest_cb(frame, 1)
		cv2.imshow("Before", frame) 
		cv2.imshow("After", out)
		cv2.waitKey(0)
	cap.release()
    cv2.destroyAllWindows()
	
