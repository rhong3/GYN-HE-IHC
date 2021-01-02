# IHC and HE alignment
import openslide
from openslide import OpenSlide
import numpy as np
from PIL import Image
import skimage.morphology as skm
from scipy.ndimage.interpolation import rotate
import pandas as pd
import os
import multiprocessing as mp
import csv


# Read original slides, get valid size at 20X, and get low resolution image at level2
def read_valid(pathtosld):
    slide = OpenSlide(pathtosld)
    upperleft = [int(slide.properties['openslide.bounds-x']),
                 int(slide.properties['openslide.bounds-y'])]
    lowerright = [int(int(slide.properties['openslide.bounds-width']) / 16),
                  int(int(slide.properties['openslide.bounds-height']) / 16)]
    x = int(slide.properties['openslide.bounds-width']) - int(slide.properties['openslide.bounds-x'])
    y = int(slide.properties['openslide.bounds-height']) - int(slide.properties['openslide.bounds-y'])
    outimg = slide.read_region(upperleft, 2, lowerright).convert('RGB')

    return outimg, x, y


# Binarize images
def binarize(img):
    img = np.array(img)[:, :, :3]
    img = np.nan_to_num(img, nan=0, posinf=0, neginf=0)
    maska = (img[:, :, :3] > 220).astype(np.uint8)
    maska = maska[:, :, 0] * maska[:, :, 1] * maska[:, :, 2]
    maskb = (img[:, :, :3] < 50).astype(np.uint8)
    maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
    maskc = (maska + maskb)

    mask = np.empty([maskc.shape[0], maskc.shape[1], 3])
    mask[:, :, 0] = maskc
    mask[:, :, 1] = maskc
    mask[:, :, 2] = maskc

    mask = (-(mask-1))
    mask = skm.binary_closing(mask)
    mask = skm.binary_dilation(mask)
    mask = skm.binary_erosion(mask)
    mask = skm.remove_small_objects(mask, min_size=300000, connectivity=1, in_place=False)
    mask = skm.remove_small_holes(mask, area_threshold=200000, connectivity=1, in_place=False)

    return mask


# Padding helper function
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# Optimization method
# angle is the rotation angle each time
# stepdecay defines how many positions to select on each side
def optimize(imga, imgb, ID, angle=30, stepdecay=10):
    rot = int(360/angle)
    imgb = imgb[:, :, 0]
    # padding size is the diagnal of IHC minus minimum side of HE +1
    pdd = int(np.sqrt(imgb.shape[0]**2+imgb.shape[1]**2)-np.amin(imga.shape[:2])+1)
    imga = imga[:, :, 0]
    imga = np.pad(imga, pdd, mode=pad_with).astype('uint8')

    # original canvas is the padded HE-sized canvas with IHC in the middle
    ori_canvas = np.zeros(imga.shape)
    ori_canvas[pdd:pdd+imgb.shape[0], pdd:pdd+imgb.shape[1]] = imgb
    # default best canvas is the original canvas
    globalmax = 0
    globalbest_canvas = ori_canvas
    globalmax_coor = [0, 0, pdd, pdd, pdd, angle, stepdecay]
    # t represents transpose; r represents rotation; i and j are coordinates
    for t in range(2):
        if t == 0:
            imgbt = imgb
        else:
            imgbt = np.transpose(imgb)
        for r in range(rot):
            imgx = rotate(imgbt, r*angle)
            # post processing after rotation
            imgx = (imgx > 0.5).astype(np.uint8)
            imgx = skm.remove_small_objects(imgx, min_size=100, connectivity=1, in_place=False)
            imgx = skm.remove_small_holes(imgx, area_threshold=100, connectivity=1, in_place=False)
            # determine start and end coordinates
            istart = 0
            jstart = 0
            iend = imga.shape[0]-imgx.shape[0]
            jend = imga.shape[1]-imgx.shape[1]
            # determine the step (width) of each selected position
            step = int(np.amax([iend, jend]) / stepdecay)
            maxx = 0
            best_coor = [t, r, istart, jstart, pdd]
            sndbest_coor = [t, r, iend, jend, pdd]
            while step >= 1:
                newmax = False
                for i in range(istart, iend, step):
                    for j in range(jstart, jend, step):
                        canvas = np.zeros(imga.shape)
                        canvas[i:i + imgx.shape[0], j:j + imgx.shape[1]] = imgx
                        canvas = canvas.astype('uint8')
                        # calculate overlap pixels
                        summ = np.sum(np.multiply(imga, canvas))
                        # determine local max os overlap
                        if summ > maxx:
                            sndbest_coor = best_coor
                            maxx = summ
                            best_coor = [t, r, i, j, pdd, angle, stepdecay]
                            best_canvas = canvas
                            newmax = True
                # determine next round coordinates based on local max and second max of previous round
                # if no new max found, move on to next rotation
                istart = np.amin([best_coor[2], sndbest_coor[2]])
                iend = np.amax([best_coor[2], sndbest_coor[2]])
                jstart = np.amin([best_coor[3], sndbest_coor[3]])
                jend = np.amax([best_coor[3], sndbest_coor[3]])
                if istart == iend:
                    iend += 1
                if jstart == jend:
                    jend += 1
                if newmax:
                    step = int(np.amax([int(iend-istart), int(jend-jstart)])/stepdecay)
                else:
                    break
            # determine if new global max found in current rotation
            if maxx > globalmax:
                globalmax = maxx
                globalmax_coor = best_coor
                globalbest_canvas = best_canvas
    # print and output global max coordinates and canvas
    print(ID, end=' ')
    print("Global max overlap: {}".format(globalmax))
    print(ID, end=' ')
    print('Global best coordinate: ', end='')
    print(globalmax_coor)

    return globalmax_coor, globalmax, globalbest_canvas, imga


# Reconstruct RGB images from binary images
def cvs_to_img(cvss):
    canvas_out = cvss * 255
    if len(canvas_out.shape) == 3:
        canvas_out_RGB = Image.fromarray(canvas_out.astype('uint8'), 'RGB')
    else:
        canvas_out_RGB = np.empty([canvas_out.shape[0], canvas_out.shape[1], 3])
        canvas_out_RGB[:, :, 0] = canvas_out
        canvas_out_RGB[:, :, 1] = canvas_out
        canvas_out_RGB[:, :, 2] = canvas_out
        canvas_out_RGB = Image.fromarray(canvas_out_RGB.astype('uint8'), 'RGB')

    return canvas_out_RGB


# visualize the overlaps of binary images
def overlap(cvsa, cvsb, coordinate):
    cvsc = (cvsa+cvsb)/2
    cvsc = cvsc[coordinate[4]:int(cvsc.shape[0]-coordinate[4]), coordinate[4]:int(cvsc.shape[1]-coordinate[4])]

    return cvsc


# lay over optimized IHC onto H&E based on best coordinates
def overslides(imga, imgb, coor):
    imga = np.array(imga)[:, :, :3]
    imgb = np.array(imgb)[:, :, :3]
    if coor[0] == 1:
        imgb = np.transpose(imgb)
    imgb = rotate(imgb, coor[1]*coor[5])
    canvasa = []
    for i in range(3):
        canvasa.append(np.pad(imga[:, :, i], coor[4], mode=pad_with).astype('uint8'))
    canvasa = np.stack(canvasa, axis=2)
    canvasb = np.zeros(canvasa.shape)
    canvasb[coor[2]:int(coor[2]+imgb.shape[0]), coor[3]:int(coor[3]+imgb.shape[1]), :] = imgb
    canvasc = np.ubyte(0.3 * canvasa + 0.7 * canvasb)
    canvasc = canvasc[coor[4]:int(coor[4]+imga.shape[0]), coor[4]:int(coor[4]+imga.shape[1]), :]
    outimg = Image.fromarray(canvasc.astype('uint8'), 'RGB')

    return outimg


# Main process method for multi-processing
def main_process(HE_File, HE_ID, IHC_File, IHC_ID):
    PID = HE_ID.split('-')[0]
    HEID = HE_ID.split('-')[1]

    tnl, tnl_x, tnl_y = read_valid('../images/NYU/{}'.format(HE_File))
    itnl, itnl_x, itnl_y = read_valid('../images/NYU/{}'.format(IHC_File))

    try:
        os.mkdir("../align/{}".format(PID))
    except FileExistsError:
        pass
    try:
        os.mkdir("../align/{}/{}".format(PID, HEID))
    except FileExistsError:
        pass
    try:
        os.mkdir("../align/{}/{}/{}".format(PID, HEID, IHC_ID))
    except FileExistsError:
        pass

    print("Now processing {} of {} ...".format(IHC_ID, PID), flush=True)
    infolist = [PID, HEID, IHC_ID, HE_File, IHC_File, tnl_x, tnl_y, itnl_x, itnl_y]

    tnl.save('../align/{}/{}/{}/he.jpg'.format(PID, HEID, IHC_ID))
    btnl = binarize(tnl)
    cvs_to_img(btnl).save('../align/{}/{}/{}/he-b.jpg'.format(PID, HEID, IHC_ID))

    itnl.save('../align/{}/{}/{}/ihc.jpg'.format(PID, HEID, IHC_ID))
    bitnl = binarize(itnl)
    cvs_to_img(bitnl).save('../align/{}/{}/{}/ihc-b.jpg'.format(PID, HEID, IHC_ID))

    coor, gmax, cvs, he_cvs = optimize(btnl, bitnl, IHC_ID, 180, 5)

    ovl = overlap(cvs, he_cvs, coor)
    cvs_to_img(ovl).save('../align/{}/{}/{}/overlap.jpg'.format(PID, HEID, IHC_ID))

    overslides(tnl, itnl, coor).save('../align/{}/{}/{}/slide_overlay.jpg'.format(PID, HEID, IHC_ID))

    infolist.extend(coor)

    with open('../align/summary.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(infolist)

    return infolist


if __name__ == '__main__':
    try:
        os.mkdir("../align")
    except FileExistsError:
        pass
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
    temp = pool.starmap(main_process, tasks)
    tempdict = list(temp)
    pool.close()
    pool.join()
    aligned = list(filter(None, tempdict))

    alignedpd = pd.DataFrame(aligned, columns=['Patient_ID', 'H&E_ID', 'IHC_ID', 'H&E_File', 'IHC_File',
                                               'H&E_X', 'H&E_Y', 'IHC_X', 'IHC_Y', 'transpose', 'rotation',
                                               'istart', 'jstart', 'padding', 'angle', 'step_decay'])
    alignedpd.to_csv('../align/final_summary.csv', header=True, index=False)

