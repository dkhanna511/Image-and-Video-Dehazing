import cv2
import glob
import os 
interval = 75
stride = 256
#save_dir = "./dataset_256_GT3/"
save_dir = "./dataset_256_hazy3/"
patch_dimension=64
count=0
def getPatches(image, height, width):
    i=0
    global count
    while (i<height):
        j=0
        while (j<width):

            if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
                rs=i
                re = i+patch_dimension
                cs = j
                ce = j+patch_dimension

            if i+patch_dimension >= height and j+patch_dimension <=width-1:
                rs = height-(patch_dimension)
                re = height
                cs = j
                ce = j+patch_dimension

            if i+patch_dimension <= height-1 and j+patch_dimension >=width:
                rs = i
                re = i+patch_dimension
                cs = width - (patch_dimension)
                ce = width

            if i+patch_dimension >= height and j+patch_dimension >=width:
                rs = height-(patch_dimension)
                re = height
                cs = width - (patch_dimension)
                ce = width

        
            cropimage = image[rs:re, cs:ce]
            count+=1
            cv2.imwrite(save_dir + str(count).zfill(5) + '.PNG', cropimage)
            print("====SAVING====" + " : " + str(count).zfill(5))
            j=j+32
        i=i+32

#image_files = sorted(glob.glob("./GT1/*.PNG"))
image_files = sorted(glob.glob("./IP1/*.PNG"))
for image_file in image_files:
    img = cv2.imread(image_file)  
    imgYCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    (Y ,Cr, Cb)=cv2.split(img)
    #print("Y shape = ", Y.shape)
    print("img shape = " , img.shape)

    #Y=Y.reshape(Y.shape[0], Y.shape[1], 1)
    print("Y shape = ", Y.shape)

    bname = os.path.basename(image_file)
    head,ext = os.path.splitext(bname)
    print(head)
    #cv2.imshow('frame' , Y)
    getPatches(Y,Y.shape[0],Y.shape[1])
