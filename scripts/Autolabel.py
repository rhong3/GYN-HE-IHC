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
    imga = np.array(imga)[:, :, :3]
    imgb = np.array(imgb)[:, :, :3]
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
    outimg = outimg.resize((outimg.size[0]*16, outimg.size[1]*16))

    return outimg


# Find if each tile is positive
def tile_test(maskk, tsize, stepsize, start_coord, thup=0.9, thlr=0.1):
    x_start = int((2000-start_coord[0] % 2000))
    y_start = int((2000-start_coord[1] % 2000))
    outlist = []
    for i in range(x_start, int(maskk.shape[0]-tsize), stepsize):
        for j in range(y_start, int(maskk.shape[1]-tsize), stepsize):
            pos_rate = round(np.sum(maskk[i:i+tsize, j:j+tsize, 0])/(tsize**2), 5)
            outlist.append([i, j, int(i+start_coord[0]), int(i+start_coord[1]), pos_rate, int(thlr < pos_rate < thup)])

    return outlist


# Main process method for multi-processing
def main_p(HE_File, PID, HEID, IHC_File, IHC_ID, *args):
    try:
        tnl, _, _, start_coor = read_valid('../images/NYU/{}'.format(HE_File))
        itnl, _, _, _ = read_valid('../images/NYU/{}'.format(IHC_File))
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
    alimg = reconstruct(tnl, itnl, args)
    alimg.save('../autolabel/{}/{}/{}/ihc-align.png'.format(PID, HEID, IHC_ID))
    almask = threshold(alimg)
    cvs_to_img(almask).save('../autolabel/{}/{}/{}/ihc-align-b.png'.format(PID, HEID, IHC_ID))
    labels = tile_test(almask, 150, 125, start_coor)
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

