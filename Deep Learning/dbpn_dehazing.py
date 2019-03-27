import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
#from utils import *
import cv2
import glob
import os 
from scipy.ndimage import imread
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


dirsave = "Results_Training_Time/"
dirdata = "../generate_dataset/"
dir_test_data = "test/"
dirsave_test = "result/"
learning_rate = 0.0001
epochs = 500
batchsize = 4
display_step = 5

dimension = 64
# n_input = dimension
# patch_dimension = 64
# n_output = n_input
# dim = n_input

ll = 0
hl = 0
incr = batchsize
# noisy_image = create_patch_normal(int(dimension/4), "IP\\", 4)
#true_image, noisy_image = create_patch_normal(dimension, "GT/", 4)
#print(np.shape(true_image))
#print(noisy_image.shape)
#plt.imshow(true_image[100])
# plt.imshow(noisy_image[100])
#test_img = create_patch_normal(dimension, "\\Test\\", 4)
# print(np.shape(true_image))
# print(noisy_image.shape)
print("====LOADING NOISY===")
noisy_image = np.load('dataset_hazy_256_1.npy')
#noisy_image=noisy_image[:100,:,:,:]
print("====LOADING TRUE IMAGES===")
true_image = np.load('dataset_GT_256_1.npy')
#true_image=true_image[:100,:,:,:]
print("LOADING COMPLETE!")
# noisy_image = np.array(noisy_image)
noisy_image = noisy_image.astype(float)
true_image = true_image.astype(float)

print("size of true image " , true_image.shape)

print("size of noisy image " , noisy_image.shape)
# test_img = test_img.astype(float)
# images = images.astype(float)

#noisy_image = normalize(255.0, 0.0, 1.0, 0.0, noisy_image)
#true_image = normalize(255.0, 0.0, 1.0, 0.0, true_image)
# test_img = normalize(255.0, 0.0, 1.0, 0.0, test_img)

'''
noisy_image = noisy_image/255
true_image=true_image/255
images = images/255
'''

# print(true_image)

totsize = true_image.shape[0]

print(" Total size of patches ", totsize)
# totalsize = test_img.shape[0]
lowerlimit = 0
higherlimit = 0



def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr



def nextbatch(batch_i):
    global ll
    global hl

    ll = batch_i * batchsize
    hl = batch_i * batchsize + (batchsize)
    #print ( "ll  == ", ll)
    #print("hl  == ", hl)

    # print hl
    # print ll
    tempnoisy = noisy_image[ll:hl].copy()

    

    #print("shape of tempnoisy is : ", tempnoisy.shape)

    tempnormal = true_image[ll:hl].copy()

    #print("shape of tempnormal is : ", tempnormal.shape)
    #tempnormal = tempnormal.reshape((batchsize, dimension, dimension, 3))
    #tempnoisy = tempnoisy.reshape((batchsize, dimension, dimension, 3))
    tempnormal = tempnormal.reshape((batchsize, dimension, dimension, 1))
    #print("shape of tempnoisy is : ", tempnoisy.shape)
    tempnoisy = tempnoisy.reshape((batchsize, dimension, dimension, 1))
    #print("shape of tempnoisy is : ", tempnoisy.shape)




    #print("tempnormal == ", tempnormal)

    #print("tempnoisy == ", tempnoisy)

    # print tempnoisy.shap
    return tempnormal, tempnoisy


# WEIGHTS AND BIASES

n1 = 8
n2 = 16
n3 = 32

ksize = 3
ksize1 = 5

with tf.device('/device:GPU:0'):
    weightsin = {
        'inthreebythree': tf.Variable(tf.random_normal([ksize, ksize, 3, n1], stddev=0.1))

    }
    biasesin = {
        'inthreebythree': tf.Variable(tf.random_normal([n1], stddev=0.1))

    }
    '''
    weightsdimred = {
        'inonebyone': tf.Variable(tf.random_normal([1, 1, n1, n3], stddev=0.1))
    '''
    weightsdimred = {
        'inonebyone': tf.Variable(tf.random_normal([1, 1, 1, n3], stddev=0.1))





    }
    biasesdimred = {
        'inonebyone': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }

    weightsrec = {
        'inonebyone': tf.Variable(tf.random_normal([1, 1, n3, 3], stddev=0.1))

    }
    biasesrec = {
        'inonebyone': tf.Variable(tf.random_normal([3], stddev=0.1))

    }

    # WEIGHTS 1
    weightsupb1 = {
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesupb1 = {
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }
    weightsdwb1 = {
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesdwb1 = {
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }

    # WEIGHTS 2

    weightsupb2 = {
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesupb2 = {
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }
    weightsdwb2 = {
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesdwb2 = {
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }

    # WEIGHTS 3
    weightsupb3 = {
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesupb3 = {
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }

    # WEIGHTS 4
    weightsupb4 = {
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesupb4 = {
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }
    weightsdwb4 = {
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesdwb4 = {
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }

    # WEIGHTS 5

    weightsupb5 = {
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesupb5 = {
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }
    weightsdwb5 = {
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesdwb5 = {
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }

    # WEIGHTS 6
    weightsupb6 = {
        'us1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'ds1': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1)),
        'us2': tf.Variable(tf.random_normal([ksize1, ksize1, n3, n3], stddev=0.1))

    }

    biasesupb6 = {
        'bus1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bds1': tf.Variable(tf.random_normal([n3], stddev=0.1)),
        'bus2': tf.Variable(tf.random_normal([n3], stddev=0.1))

    }


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def caeUSB(_X, _W, _b, _keepprob, alpha=0.2):
    _input_r = _X
    print( " =====UPSAMPLING=====")
    _h0 = tf.add(tf.nn.conv2d_transpose(_input_r, _W['us1'], tf.stack(
        [tf.shape(_input_r)[0], tf.shape(_input_r)[1] * 2, tf.shape(_input_r)[2] * 2, n3]), strides=[1, 2, 2, 1],
                                        padding='SAME'), _b['bus1']) 

    #print("_h0 ==  ", _h0.shape)


    _ch0 = leaky_relu(_h0)
    _ch0 = tf.nn.dropout(_ch0, _keepprob)

    _l0 = tf.add(tf.nn.conv2d(_ch0, _W['ds1'], strides=[1, 2, 2, 1], padding='SAME'), _b['bds1'])
    _cl0 = leaky_relu(_l0)
    _cl0 = tf.nn.dropout(_cl0, _keepprob)

    #print("_cl0 == ", _cl0.shape )

    _e0 = _cl0 - _input_r

    _h1 = tf.add(tf.nn.conv2d_transpose(_e0, _W['us2'], tf.stack(
        [tf.shape(_input_r)[0], tf.shape(_input_r)[1] * 2, tf.shape(_input_r)[2] * 2, n3]), strides=[1, 2, 2, 1],
                                        padding='SAME'), _b['bus2'])
    _ch1 = leaky_relu(_h1)
    _ch1 = tf.nn.dropout(_ch1, _keepprob)

    _ht = _ch0 + _ch1
    _out = _ht

    print(_out.shape)
    #print("_out == ", _out.shape)
    return _out


def caeDSB(_X, _W, _b, _keepprob, alpha=0.2):
    _input_r = _X
    print(" =====DOWNSAMPLING=====")
    _l0 = tf.add(tf.nn.conv2d(_input_r, _W['ds1'], strides=[1, 2, 2, 1], padding='SAME'), _b['bds1'])
    _cl0 = leaky_relu(_l0)
    _cl0 = tf.nn.dropout(_cl0, _keepprob)

    #print("_cl0 == ", _cl0.shape)

    _h0 = tf.add(tf.nn.conv2d_transpose(_cl0, _W['us1'], tf.stack(
        [tf.shape(_input_r)[0], tf.shape(_input_r)[1], tf.shape(_input_r)[2], n3]), strides=[1, 2, 2, 1],
                                        padding='SAME'), _b['bus1'])
    _ch0 = leaky_relu(_h0)
    _ch0 = tf.nn.dropout(_ch0, _keepprob)

    #print("_ch0 == ", _ch0)

    _e0 = _ch0 - _input_r

    _l1 = tf.add(tf.nn.conv2d(_e0, _W['ds2'], strides=[1, 2, 2, 1], padding='SAME'), _b['bds2'])
    _cl1 = leaky_relu(_l1)
    _cl1 = tf.nn.dropout(_cl1, _keepprob)

    _lt = _cl0 + _cl1
    _out = _lt
    print(_out.shape)

    #print("_out ==")

    return _out


def calculateL2loss(im1, im2):
    return tf.reduce_mean(tf.square(im1 - im2))


def calculateL1loss(im1, im2):
    return tf.reduce_sum(tf.abs(im1 - im2))


def optimize(cost, learning_rate=0.0001):
    return tf.train.AdamOptimizer(learning_rate).minimize(cost)

x = tf.placeholder(tf.float32, [None, None, None, 1])
y = tf.placeholder(tf.float32, [None, None, None, 1])

keepprob = tf.placeholder(tf.float32)

#op_l1 = conv2d(x, weightsin['inthreebythree'], biasesin['inthreebythree'])
#print(op_l1.shape)

#op_l2 = conv2d(op_l1, weightsdimred['inonebyone'], biasesdimred['inonebyone'])
#print(op_l2.shape)

op_l2 = conv2d(x, weightsdimred['inonebyone'], biasesdimred['inonebyone'])


op_ub1 = caeUSB(op_l2, weightsupb1, biasesupb1, keepprob)
#print(op_ub1.shape)

op_db1 = caeDSB(op_ub1, weightsdwb1, biasesdwb1, keepprob)
#print(op_db1.shape)

op_ub2 = caeUSB(op_db1, weightsupb2, biasesupb2, keepprob)
#print(op_ub2.shape)

op_db2 = caeDSB(op_ub2, weightsdwb2, biasesdwb2, keepprob)
#print(op_db2.shape)

op_ub3 = caeUSB(op_db2, weightsupb3, biasesupb3, keepprob)
#print(op_ub3.shape)
#op_ub4 = caeUSB(op_ub3, weightsupb4, biasesupb4, keepprob)

op_db4 = caeDSB(op_ub3, weightsdwb4, biasesdwb4, keepprob)
#print(op_db4.shape)

op_ub5 = caeUSB(op_db4, weightsupb5, biasesupb5, keepprob)
#print(op_ub5.shape)

op_db5 = caeDSB(op_ub5, weightsdwb5, biasesdwb5, keepprob)


print(op_db5.shape)
#print(op_db5.shape)
#op_ub6 = caeUSB(op_db5, weightsupb6, biasesupb6, keepprob)

pred = conv2d(op_db5, weightsrec['inonebyone'], biasesrec['inonebyone'])

cost1 = calculateL2loss(pred, y)

optm1 = optimize(cost1, learning_rate)

print("Network ready")

init = tf.global_variables_initializer()

print("All functions ready")
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

print("Start training")

batch_xs_noisy = []
batch_xs = []

for epoch_i in range(epochs):
    print("===EPOCH TRAINING=== : "+str(epoch_i))
    num_batch = int(totsize / (batchsize))
    for batch_i in range(num_batch):
        batch_xs, batch_xs_noisy = nextbatch(batch_i)
        _,cost=sess.run([optm1, cost1], feed_dict={x: batch_xs_noisy, y: batch_xs, keepprob: 1})

    ll = 0
    hl = 0
    #cost = sess.run(cost1, feed_dict={x: batch_xs_noisy, y: batch_xs, keepprob: 1})
    print("[%02d/%02d] Cost1: %.6f" % (epoch_i, epochs, cost))
    if cost < 0.0005:
        learning_rate = 0.00001
    elif cost < 0.0002:
        learning_rate = 0.000001
    else:
    	learning_rate = 0.0001

    if epoch_i % display_step == 0 or epoch_i == epochs - 1:
        saver.save(sess, "logs/weights_dehaze.ckpt")
        for img_from_folder in sorted(glob.glob(dir_test_data + "*PNG")):
            img = cv2.imread(img_from_folder)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            (Y, Cr, Cb) = cv2.split(img)

            print(img.shape)

            print(Y.shape)


            #img = img / float(255)
            Y = Y / float(255)


            fname = img_from_folder
            fname = os.path.basename(fname)
            #img = img.reshape(1,img.shape[0],img.shape[1],3)

            Y = Y.reshape(1,Y.shape[0],Y.shape[1],1)
            


            #patch = sess.run(pred, feed_dict={x: img, keepprob: 1.})
            
            patch = sess.run(pred, feed_dict={x: Y, keepprob: 1.})
            


            recon = cv2.multiply(patch,255)



            print(patch.shape)

            #recon = recon.reshape(recon.shape[1],recon.shape[2],3)
            result=cv2.merge((recon, Cr, Cb))
            recon = recon.reshape(recon.shape[1],recon.shape[2],3)
            

            filename=dirsave_test + str(epoch_i) + ("_") + fname



            #cv2.imwrite(dirsave_test + str(epoch_i) + ("_") + fname, recon)
            cv2.imwrite(filename, recon)
            #result=output_psnr_mse(img_from_folder)
        print("=========SAVED TEST RESULTS=======")
            #in_arr = getPatches_org(img, img.shape[0], img.shape[1], in_arr, int(dimension/2), 0)
            #cv2.imshow('image', in_arr[200])
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            # for j, patches in enumerate(in_arr):
            #     patches = np.reshape([patches], (1,int(dimension/2), int(dimension/2),1))
            #     patch = sess.run(pred, feed_dict={x: patches, keepprob: 1.})
            #     patch = np.reshape([patch], (dimension, dimension))
            #     out_arr.append(patch)
            # recon = takeAllPatches(out_arr, img.shape[1]*2, img.shape[0]*2, dimension, 0)
            #cv2.imshow('image', recon)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #recon = cv2.multiply(recon,255)
            #recon = recon.astype('uint8')
            #cv2.imwrite(dirsave_test + str(epoch_i) + ("_") + fname, recon)
        '''
        for i in range(countperimage.shape[0]):
            higherlimit = int(higherlimit+countperimage[i])
            allpatchesofanimage = images[lowerlimit:higherlimit].copy()
            lowerlimit = int(lowerlimit + countperimage[i])
            print countperimage[i]
            reconstructedimage = np.zeros([int(countperimage[i]), patch_dimension, patch_dimension])
            for j in range(int(countperimage[i])):
                recon = sess.run(pred, feed_dict = {x:allpatchesofanimage[j].reshape(1, patch_dimension/4, patch_dimension/4, 1), keepprob:1.})
                recon = recon.reshape((1, patch_dimension, patch_dimension))
                reconstructedimage[j]=recon

            recreatedimage = takeAllPatches(reconstructedimage, int(widthofimages[i]*4), int(heightofimages[i]*4))
            recreatedimage = normalize(1.0, 0.0, 255.0, 0.0, recreatedimage)
            cv2.imwrite(dirsave+names[i], recreatedimage)
        lowerlimit=0
        higherlimit=0
        '''
