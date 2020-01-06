## To compare with the netvlad code
import keras
import json
import pprint
import numpy as np
import cv2
import code
import glob
from numpy import array
import os
import TerminalColors
tcol = TerminalColors.bcolors()

from CustomNets import NetVLADLayer, GhostVLADLayer
from predict_utils import open_json_file, change_model_inputshape

def init_net():
    """
    The way to a)load json-formatted models b)loading weights c) sample prediction
    """
    # LOG_DIR = 'models.keras/Apr2019/K16_gray_training/'
    LOG_DIR = '/app/datasets/models.keras/color_conv6_K16Ghost1__centeredinput/'
    # core_model.1000.keras

    # Load JSON formatted model
    json_string = open_json_file( LOG_DIR+'/model.json' )
    print '======================='
    pprint.pprint( json_string, indent=4 )
    print '======================='
    model = keras.models.model_from_json(str(json_string),  custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer} )
    print 'OLD MODEL: '
    model.summary()
   # quit()

    # Load Weights from model-file
    model_fname = LOG_DIR+'/core_model.%d.keras' %(2000)
    print 'Load model: ', model_fname
    model.load_weights(  model_fname )
    return model 
def init_net_hdf5():
    kerasmodel_file = '/app/datasets/models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenet-conv_pw_6_relu__K16__allpairloss/modelarch_and_weights.700.h5'
    assert os.path.isfile( kerasmodel_file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_file

    #-----
    # Load from HDF5
    print tcol.OKGREEN, 'Load model: ', kerasmodel_file, tcol.ENDC
    model = keras.models.load_model(  kerasmodel_file, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer':GhostVLADLayer} )
    old_input_shape = model._layers[0].input_shape
    print 'OLD MODEL: ', 'input_shape=', str(old_input_shape)

    return model


def model_config(model, im_w, im_h):    # Replace Input Layer
    new_input_shape=(1,im_w,im_h,3)   #(1,480,640,1)
    new_model = change_model_inputshape( model, new_input_shape=new_input_shape )
    return new_model 


def processFeatImg(new_model, X ):
    # Sample Predict
    # test new model on a random input image

    #load all images and change the range from 0-255 to -.5 - .5 ## ((:-128)/255)
    # X = np.random.rand(new_input_shape[0], new_input_shape[1], new_input_shape[2], new_input_shape[3] )
    X = (X.astype('float32')-128. )*2.0/255.
    # code.interact( local=locals() )

    if len(X.shape) == 2:
        X = np.expand_dims( np.expand_dims( X, -1 ), 0 )
    
    # code.interact( local=locals() )
    if len(X.shape) == 3:
        X = np.expand_dims( X, 0 )


    y_pred = new_model.predict(X)
    print('try predict with a random input_img with shape='+str(X.shape)+ str(y_pred) )
    return y_pred

def db247():
    # main_model = init_net() 
    main_model = init_net_hdf5() 
    new_model_640_480 = model_config(main_model,480,640)
    new_model_360_480 = model_config(main_model,480,360)
    new_model_360_640 = model_config(main_model,360,640)

    Y = [] 
    dataset_DIR = '/app/datasets/NetvLad/247_Tokyo_GSV/query/query_manohar/'
    #dataset_DIR = '/app/datasets/NetvLad/247_Tokyo_GSV/images/original_monohar/'
    for filename in glob.glob(dataset_DIR+'/*.jpg'):
        print filename
       # im = cv2.imread(filename,0)
        im = cv2.imread(filename)
        #cv2.imshow('windowname', im)
        #cv2.waitKey(10)
        # im_w, im_h, im_c = im.shape
        # print 'Width of the image is %d, and height is %d and channel is %d', im_w, im_h, im_c

        im_w, im_h = im.shape[0:2]
        print 'Width of the image is %d, and height is %d', im_w, im_h
        if im_w == 480 and im_h == 640:
            y = processFeatImg(new_model_640_480, im) # y.shape = (1, 4096)
        elif im_w == 480 and im_h == 360:
             y = processFeatImg(new_model_360_480, im) # y.shape = (1, 4096)
        elif im_w == 360 and im_h == 640:
             y = processFeatImg(new_model_360_640, im) # y.shape = (1, 4096)
        y = y.flatten()
        Y.append(y)

    return Y    #(216, 4096) (total 216 images, each has 4096 lenght of descriptor)

    # Save to bin file 
 
def savefile(b_vector, filename):
    #a = array(floats,'float32')
    output_file = open(filename, 'wb')
    b_vector.tofile(output_file)
    output_file.close()

if __name__ == '__main__':
    yy = db247()    # len(yy) =  216
    xy = np.asarray(yy)
    # xy = xy.transpose()
    xy = xy.flatten() # len(xy) = 884736
    print len(xy)
    # to check the norm `np.linalg.norm(xy[:4096])`
    
    #savefile(xy,'db247.bin')
    savefile(xy,'q247.bin')
    #Now save this to bin file.

# if __name__ == '__main__':
#     main_model = init_net() 
#     new_model_640_480 = model_config(main_model,480,640)
#     im = cv2.imread("/app/datasets/NetvLad/247_Tokyo_GSV/query/00001.jpg", 0)
#     import code 
#     y = processFeatImg(new_model_640_480, im )
#     code.interact( local=locals() )
    