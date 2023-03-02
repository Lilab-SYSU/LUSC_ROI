#-*-coding:utf-8-*-
import numpy as np
import time
import os

import openslide
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Pool
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
import sys
sys.path.append('/home/myang/PycharmProjects/PyTorch_test/Pathology_WSI/LILI/predict')

from Estimate_W import Wfast, suppress_stdout

def ColorNorm(source,target, nstains=2, lamb=0.01, output_direc="./", img_level=0, background_correction=True, config=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if config is None:
        config = tf.ConfigProto(log_device_placement=False)

    g_1 = tf.Graph()
    with g_1.as_default():
        Wis1 = tf.placeholder(tf.float32)
        Img1 = tf.placeholder(tf.float32, shape=(None, None, 3))
        src_i_0 = tf.placeholder(tf.float32)

        s = tf.shape(Img1)
        Img_vecd = tf.reshape(tf.minimum(Img1, src_i_0), [s[0] * s[1], s[2]])
        V = tf.log(src_i_0 + 1.0) - tf.log(Img_vecd + 1.0)
        Wi_inv = tf.transpose(tf.py_func(np.linalg.pinv, [Wis1], tf.float32))
        Hiv1 = tf.nn.relu(tf.matmul(V, Wi_inv))

        Wit1 = tf.placeholder(tf.float32)
        Hiv2 = tf.placeholder(tf.float32)
        sav_name = tf.placeholder(tf.string)
        tar_i_0 = tf.placeholder(tf.float32)
        normfac = tf.placeholder(tf.float32)
        shape = tf.placeholder(tf.int32)

        Hsonorm = Hiv2 * normfac
        source_norm = tf.cast(tar_i_0 * tf.exp((-1) * tf.reshape(tf.matmul(Hsonorm, Wit1), shape)), tf.uint8)
        enc = tf.image.encode_png(source_norm)  ###anntation by mingleiyang normaled patch encode png
        fwrite = tf.write_file(sav_name, enc)

    session1 = tf.Session(graph=g_1, config=config)
    ###target init weight
    s="normal_test.png"
    if background_correction:
        correc = "back-correc"
    else:
        correc = "no-back-correc"
    tic = time.time()
    level = img_level
    print()
    file_no = 0
    if file_no == 0:
        I = openslide.open_slide(target)
        if img_level >= I.level_count:
            print("Level", img_level, "unavailable for image, proceeding with level 0")
            level = 0
        else:
            level = img_level
        xdim, ydim = I.level_dimensions[level]
        ds = I.level_downsamples[level]

        if file_no == 0:
            print("Target Stain Separation in progress:", target, str(xdim) + str("x") + str(ydim))
        else:
            print("Source Stain Separation in progress:", target, str(xdim) + str("x") + str(ydim))
        print("\t \t \t \t \t \t \t \t \t \t Time: 0")

        # parameters for W estimation
        num_patches = 100
        patchsize = 256  # length of side of square

        i0_default = np.array([255., 255., 255.], dtype=np.float32)

        Wi, i0 = Wfast(I, nstains, lamb, num_patches, patchsize, level, background_correction)
        if i0 is None:
            print("No white background detected")
            i0 = i0_default

        if not background_correction:
            print("Background correction disabled, default background intensity assumed")
            i0 = i0_default

        if Wi is None:
            print("Color Basis Matrix Estimation failed...image normalization skipped")
            # continue
        print("W estimated", )
        print("\t \t \t \t \t \t Time since processing started:", round(time.time() - tic, 3))
        Wi = Wi.astype(np.float32)

        if file_no == 0:
            global Wi_target,tar_i0
            Wi_target = np.transpose(Wi)
            tar_i0 = i0
            print("Target Image Background white intensity:", i0)
        else:
            print("Source Image Background white intensity:", i0)

        _max = 2000
        # raise valueError()
        print()
        if (xdim * ydim) <= (_max * _max):
            print("Small image processing...")
            img = np.asarray(I.read_region((0, 0), level, (xdim, ydim)), dtype=np.float32)[:, :,
                  :3]  ##by mingleiyang np

            Hiv = session1.run(Hiv1, feed_dict={Img1: img, Wis1: Wi, src_i_0: i0})
            # Hta_Rmax = np.percentile(Hiv,q=99.,axis=0)
            H_Rmax = np.ones((nstains,), dtype=np.float32)
            for i in range(nstains):
                t = Hiv[:, i]
                H_Rmax[i] = np.percentile(t[t > 0], q=99., axis=0)

            if file_no == 0:
                file_no += 1
                global Hta_Rmax
                Hta_Rmax = np.copy(H_Rmax)
        else:
            _maxtf=2550#changed from initial 3000
            x_max=xdim
            y_max=min(max(int(_maxtf*_maxtf/x_max),1),ydim)
            print ("Large image processing...")
            if file_no==0:
                Hivt=np.memmap('H_target', dtype='float32', mode='w+', shape=(xdim*ydim,2))
            else:
                Hivs=np.memmap('H_source', dtype='float32', mode='w+', shape=(xdim*ydim,2))
                sourcenorm=np.memmap('wsi', dtype='uint8', mode='w+', shape=(ydim,xdim,3))
            x_tl = range(0,xdim,x_max)
            y_tl = range(0,ydim,y_max)
            print ("WSI divided into",str(len(x_tl))+"x"+str(len(y_tl)))
            count=0
            print ("Patch-wise H calculation in progress...")
            ind=0
            perc=[]
            for x in x_tl:
                for y in y_tl:
                    count+=1
                    xx=min(x_max,xdim-x)
                    yy=min(y_max,ydim-y)
                    print ("Processing:",count,"        patch size",str(xx)+"x"+str(yy),)
                    print ("\t \t Time since processing started:",round(time.time()-tic,3))
                    img=np.asarray(I.read_region((int(ds*x),int(ds*y)),level,(xx,yy)),dtype=np.float32)[:,:,:3]

                    Hiv = session1.run(Hiv1,feed_dict={Img1:img, Wis1: Wi,src_i_0:i0})
                    if file_no==0:
                        Hivt[ind:ind+len(Hiv),:]=Hiv
                        _Hta_Rmax = np.ones((nstains,),dtype=np.float32)
                        for i in range(nstains):
                            t = Hiv[:,i]
                            _Hta_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)
                        perc.append([_Hta_Rmax[0],_Hta_Rmax[1]])
                        ind+=len(Hiv)
                        continue
                    else:
                        Hivs[ind:ind+len(Hiv),:]=Hiv
                        _Hso_Rmax = np.ones((nstains,),dtype=np.float32)
                        for i in range(nstains):
                            t = Hiv[:,i]
                            _Hso_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)
                        perc.append([_Hso_Rmax[0],_Hso_Rmax[1]])
                        ind+=len(Hiv)

            if file_no==0:
                print ("Target H calculated",)
                Hta_Rmax = np.percentile(np.array(perc),50,axis=0)
                file_no+=1
                del Hivt
                print ("\t \t \t \t \t Time since processing started:",round(time.time()-tic,3))
                ind=0

    print('source begin normal.....')
    I = source
    xdim, ydim = I.size
    num_patches = 100
    patchsize = 256  # length of side of square

    i0_default = np.array([255., 255., 255.], dtype=np.float32)

    Wi, i0 = Wfast(I, nstains, lamb, num_patches, patchsize, level, background_correction)
    if i0 is None:
        print("No white background detected")
        i0 = i0_default

    if not background_correction:
        print("Background correction disabled, default background intensity assumed")
        i0 = i0_default

    if Wi is None:
        print("Color Basis Matrix Estimation failed...image normalization skipped")
        # continue
    print("W estimated", )
    print("\t \t \t \t \t \t Time since processing started:", round(time.time() - tic, 3))
    Wi = Wi.astype(np.float32)

    print("Source Image Background white intensity:", i0)

    _max = 2000
    # raise valueError()
    print()
    if (xdim * ydim) <= (_max * _max):
        print("Small image processing...")
        img = np.asarray(I, dtype=np.float32)[:, :,
              :3]  ##by mingleiyang np

        Hiv = session1.run(Hiv1, feed_dict={Img1: img, Wis1: Wi, src_i_0: i0})
        # Hta_Rmax = np.percentile(Hiv,q=99.,axis=0)
        H_Rmax = np.ones((nstains,), dtype=np.float32)
        for i in range(nstains):
            t = Hiv[:, i]
            H_Rmax[i] = np.percentile(t[t > 0], q=99., axis=0)

        # if file_no == 0:
        #     file_no += 1
        #     Hta_Rmax = np.copy(H_Rmax)
        #     # print ("Target H calculated",)
        #     # print ("\t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3))
        #     # display_separator()
        #     continue

        print("Color Normalization in progress...")

        norm_fac = np.divide(Hta_Rmax, H_Rmax).astype(np.float32)

        '''by mingleiyang Get source normaled numpy array'''
        # sourcenorm[y:y + yy, x:x + xx, :3] = session1.run(source_norm,
        #                                                   feed_dict={Hiv2: np.array(Hivs[ind:ind + pix, :]),
        #                                                              Wit1: Wi_target, normfac: _normfac, shape: sh,
        #                                                              tar_i_0: tar_i0})

        # session1.run(fwrite, feed_dict={shape: np.array(img.shape), Wit1: Wi_target, Hiv2: Hiv, sav_name: s,
                                        # tar_i_0: tar_i0, normfac: norm_fac})
        sourceNorm=session1.run(source_norm, feed_dict={shape: np.array(img.shape), Wit1: Wi_target, Hiv2: Hiv,
                                        tar_i_0: tar_i0, normfac: norm_fac})

        # session1.run(fwrite, feed_dict={sav_name: s})

        print("File written to:", s)
        print("\t \t \t \t \t \t \t \t \t \t Total Time:", round(time.time() - tic, 3))
        # display_separator()
    session1.close()
    return sourceNorm

def main():
    nstains = 2  # number of stains
    lamb = 0.01  # default value sparsity regularization parameter
    level = 0
    df_size_min = 90
    background_correction = True
    output_direc = './'
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
    config = tf.ConfigProto(log_device_placement=False)# gpu_options=gpu_options)

    slide = openslide.OpenSlide('/public5/lilab/student/myang/project/lili/GG/test/TCGA-G3-A25Y-01Z-00-DX1.C6BF2202-9030-4460-B0F5-E846C8A44C1E.svs')
    target = '/public5/lilab/student/myang/project/lili/GG/N02171-A.kfb.mask.Tumor/59712_45621.jpg'
    source = slide.read_region((62189, 15683),0,(224,224)).convert('RGB')
    import matplotlib.pyplot as plt
    plt.imshow(source)
    plt.show()
    sourceNorm=ColorNorm(source,target,nstains,lamb,output_direc,level,background_correction,config)

    plt.imshow(sourceNorm)
    plt.show()
if __name__ == "__main__":
    main()