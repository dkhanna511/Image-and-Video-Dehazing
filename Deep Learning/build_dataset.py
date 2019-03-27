import os
import sys
import numpy as np
from PIL import Image, ImageOps

def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)

    if filename.endswith('.PNG'):
        im = im.convert('YCbCr')
    #downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(im, dtype=np.float32) / 255.


    norm_Y, norm_Cb, norm_Cr= norm_im.T
    print(norm_Y.shape)
    print("Shape of norm_Y is ", norm_Y.shape)


    #print(norm_Y.shape)
    norm_Y=norm_Y.reshape(64,  64, 1)
    print("Shape of norm_Y is ", norm_Y.shape)
    
    print("Processing" + filename)
    #downsampled_im.close()
    #im_x = np.array(im)
    im.close()
    return norm_Y
    #return im_x

if __name__ == '__main__':
    names = []
    ext = ".PNG"
    for name in sorted(os.listdir(sys.argv[1])):
         if name.endswith(ext):
            names.append((name[:-4]))
            #names.append(name)


    #dataset_X = np.zeros((len(names), 64, 64, 3))
    #dataset_Y = np.zeros((len(names), 64,64, 3))

    #dataset_X = np.zeros((len(names), 64, 64, 1))
    dataset_Y = np.zeros((len(names), 64,64, 1))


    for i in range(len(names)):
        #dataset_X[i] = _image_preprocessing(os.path.join(sys.argv[1], names[i] + '.PNG'), 32, 32)
        dataset_Y[i] = _image_preprocessing(os.path.join(sys.argv[1], names[i] + '.PNG'),32,32)
    #np.save('denoise_train_dataset_x.npy', dataset_X)
    print("=====SAVING======")
    np.save('dataset_hazy_256_1', dataset_Y)
    #np.save('dataset_GT_256_1', dataset_X)
