# Gynecological cancers H&E IHC project at NYU
# Auto-labeling
from Alignment import *
from openslide import OpenSlide
import numpy as np
from PIL import Image
import skimage.morphology as skm
from scipy.ndimage.interpolation import rotate
import pandas as pd
import os
import multiprocessing as mp
import csv
import warnings
warnings.filterwarnings("ignore")


# BEST RATIO TESTED: 10%
# Threshold images
def threshold(img):
    img = np.array(img)[:, :, :3]
    img = np.nan_to_num(img, nan=0, posinf=0, neginf=0)
    maska = (img[:, :, :3] > 160).astype(np.uint8)
    maska = maska[:, :, 0] * maska[:, :, 1] * maska[:, :, 2]
    maskb = (img[:, :, :3] < 50).astype(np.uint8)
    maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
    maskc = (maska + maskb)

    mask = np.empty([maskc.shape[0], maskc.shape[1], 3])
    mask[:, :, 0] = maskc
    mask[:, :, 1] = maskc
    mask[:, :, 2] = maskc

    mask = (-(mask-1))

    return mask


def reconstruct(imga, imgb, coor):
    if coor[0] == 1:
        imgb = np.transpose(imgb, (1, 0, 2))
    imgb = rotate(imgb, coor[1]*coor[5])
    canvasa = []
    for i in range(3):
        canvasa.append(np.pad(imga[:, :, i], coor[4], mode=pad_with).astype('uint8'))
    canvasa = np.stack(canvasa, axis=2)
    canvasb = np.zeros(canvasa.shape)
    canvasb[coor[2]:int(coor[2]+imgb.shape[0]), coor[3]:int(coor[3]+imgb.shape[1]), :] = imgb
    canvasb = canvasb[coor[4]:int(coor[4]+imga.shape[0]), coor[4]:int(coor[4]+imga.shape[1]), :]
    outimg = Image.fromarray(canvasb.astype('uint8'), 'RGB')

    return outimg, canvasb.astype('uint8')


# Find if each tile is positive
def tile_test(maskk, tsize, stepsize, th=0.1):
    outlist = []
    for i in range(0, int(maskk.shape[0]-tsize), stepsize):
        for j in range(0, int(maskk.shape[1]-tsize), stepsize):
            pos_rate = round(np.sum(maskk[i:i+tsize, j:j+tsize, 0])/(tsize**2), 5)
            outlist.append([i, j, pos_rate, int(pos_rate > th)])

    return outlist


# Main process method for multi-processing
def main_p(HE_File, PID, HEID, IHC_File, IHC_ID, *args):
    try:
        tnl, _, _ = read_valid('../images/NYU/{}'.format(HE_File))
        itnl, _, _ = read_valid('../images/NYU/{}'.format(IHC_File))
    except Exception as e:
        print('ERROR IN READING IMAGE {}'.format(IHC_ID))
        print(e)
        return []
    try:
        os.mkdir("../autolabel/{}".format(PID))
    except FileExistsError:
        pass
    try:
        os.mkdir("../autolabel/{}/{}".format(PID, HEID))
    except FileExistsError:
        pass
    try:
        os.mkdir("../autolabel/{}/{}/{}".format(PID, HEID, IHC_ID))
    except FileExistsError:
        pass

    itnl.save('../autolabel/{}/{}/{}/ihc.jpg'.format(PID, HEID, IHC_ID))
    bitnl = threshold(itnl)
    cvs_to_img(bitnl).save('../autolabel/{}/{}/{}/ihc-b.png'.format(PID, HEID, IHC_ID))
    alimg, almask = reconstruct(tnl, bitnl, args)
    alimg.save('../autolabel/{}/{}/{}/ihc-align.png'.format(PID, HEID, IHC_ID))
    labels = tile_test(almask, 150, 125)
    labels_pd = pd.DataFrame(labels, columns=['x', 'y', 'ratio', 'label'])
    labels_pd.to_csv('../autolabel/{}/{}/{}/ratio.csv'.format(PID, HEID, IHC_ID), index=False)


if __name__ == '__main__':
    ref = pd.read_csv('../align/final_summary.csv', header=0)
    # create multiporcessing pool
    print(mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())
    tasks = []
    for idx, row in ref.iterrows():
        if os.path.exists('../images/NYU/{}'.format(row['H&E_File'])) \
                and os.path.exists('../images/NYU/{}'.format(row['IHC_File'])):
            tasks.append(tuple((row['H&E_File'], row['Patient_ID'], row['H&E_ID'], row['IHC_File'], row['IHC_ID'],
                                row['transpose'], row['rotation'], row['istart'], row['jstart'], row['padding'],
                                row['angle'], row['step_decay'])))
        else:
            print('{} and {} paired files not found: {} and {}'.format(row['H&E_ID'], row['IHC_ID'],
                                                                       row['H&E_File'], row['IHC_File']))

    # process images with multiprocessing
    pool.starmap(main_p, tasks)

