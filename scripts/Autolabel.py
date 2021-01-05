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


# Main process method for multi-processing
def main_(HE_File, HE_ID, IHC_File, IHC_ID):
    PID = HE_ID.split('-')[0]
    HEID = HE_ID.split('-')[1]
    try:
        itnl, itnl_x, itnl_y = read_valid('../images/NYU/{}'.format(IHC_File))
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
    print(IHC_ID)
    bitnl= threshold(itnl)
    cvs_to_img(bitnl).save('../autolabel/{}/{}/{}/ihc-b.png'.format(PID, HEID, IHC_ID))


if __name__ == '__main__':
    ref = pd.read_csv('../NYU/align.csv', header=0)
    # create multiporcessing pool
    print(mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())
    tasks = []
    for idx, row in ref.iterrows():
        if os.path.exists('../images/NYU/{}'.format(row['H&E_File'])) \
                and os.path.exists('../images/NYU/{}'.format(row['IHC_File'])):
            tasks.append(tuple((row['H&E_File'], row['H&E_ID'], row['IHC_File'], row['IHC_ID'])))
        else:
            print('{} and {} paired files not found: {} and {}'.format(row['H&E_ID'], row['IHC_ID'],
                                                                       row['H&E_File'], row['IHC_File']))

    # process images with multiprocessing
    pool.starmap(main_, tasks)

