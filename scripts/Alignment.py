# IHC and HE alignment
from openslide import OpenSlide
import numpy as np
from PIL import Image
import skimage.morphology as skm
from scipy.ndimage.interpolation import rotate


def read_valid(pathtosld, tp):
    slide = OpenSlide(pathtosld)
    upperleft = [int(slide.properties['openslide.bounds-x']),
                 int(slide.properties['openslide.bounds-y'])]
    lowerright = [int(int(slide.properties['openslide.bounds-width']) / 16),
                  int(int(slide.properties['openslide.bounds-height']) / 16)]
    x = int(slide.properties['openslide.bounds-width']) - int(slide.properties['openslide.bounds-x'])
    y = int(slide.properties['openslide.bounds-height']) - int(slide.properties['openslide.bounds-y'])
    print("{} valid size: ".format(tp))
    print([x, y])
    outimg = slide.read_region(upperleft, 2, lowerright).convert('RGB')

    return outimg


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
    mask = skm.remove_small_objects(mask, min_size=400000, connectivity=1, in_place=False)
    mask = skm.remove_small_holes(mask, area_threshold=200000, connectivity=1, in_place=False)

    return mask


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def optimize(imga, imgb, angle=30):
    rot = int(360/angle)
    imgb = imgb[:, :, 0]
    pdd = int(np.sqrt(imgb.shape[0]**2+imgb.shape[1]**2)-np.amin(imga.shape[:2])+1)
    imga = imga[:, :, 0]
    imga = np.pad(imga, pdd, mode=pad_with).astype('uint8')

    ori_canvas = np.zeros(imga.shape)
    ori_canvas[pdd:pdd+imgb.shape[0], pdd:pdd+imgb.shape[1]] = imgb

    globalmax = 0
    globalbest_canvas = ori_canvas
    globalmax_coor = [0, 0, pdd, pdd, pdd]
    for t in range(2):
        if t == 0:
            imgbt = imgb
        else:
            imgbt = np.transpose(imgb)
        for r in range(rot):
            imgx = rotate(imgbt, r*angle)
            imgx = (imgx > 0.5).astype(np.uint8)
            imgx = skm.remove_small_objects(imgx, min_size=100, connectivity=1, in_place=False)
            imgx = skm.remove_small_holes(imgx, area_threshold=100, connectivity=1, in_place=False)
            istart = 0
            jstart = 0
            iend = imga.shape[0]-imgx.shape[0]
            jend = imga.shape[1]-imgx.shape[1]
            step = int(np.amax([iend, jend]) / 5)
            maxx = 0
            best_coor = [t, r, istart, jstart, pdd]
            sndbest_coor = [t, r, iend, jend, pdd]
            while step >= 1:
                newmax = False
                print("step size: {}".format(step))
                for i in range(istart, iend, step):
                    for j in range(jstart, jend, step):
                        canvas = np.zeros(imga.shape)
                        canvas[i:i + imgx.shape[0], j:j + imgx.shape[1]] = imgx
                        canvas = canvas.astype('uint8')

                        summ = np.sum(np.multiply(imga, canvas))
                        if summ > maxx:
                            sndbest_coor = best_coor
                            maxx = summ
                            best_coor = [t, r, i, j, pdd]
                            best_canvas = canvas
                            newmax = True
                istart = np.amin([best_coor[2], sndbest_coor[2]])
                iend = np.amax([best_coor[2], sndbest_coor[2]])
                jstart = np.amin([best_coor[3], sndbest_coor[3]])
                jend = np.amax([best_coor[3], sndbest_coor[3]])
                if istart == iend:
                    iend += 1
                if jstart == jend:
                    jend += 1
                if newmax:
                    step = int(np.amax([int(iend-istart), int(jend-jstart)])/5)
                else:
                    break
            if maxx > globalmax:
                globalmax = maxx
                globalmax_coor = best_coor
                globalbest_canvas = best_canvas
                print("new global max overlap: {}".format(globalmax))
                print('new global best coordinate: ')
                print(globalmax_coor)

    print("Global max overlap: {}".format(globalmax))
    print('Global best coordinate: ')
    print(globalmax_coor)

    return globalmax_coor, globalmax, globalbest_canvas, imga


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


def overlap(cvsa, cvsb, coordinate):
    cvsc = (cvsa+cvsb)/2
    cvsc = cvsc[coordinate[4]:int(cvsc.shape[0]-coordinate[4]), coordinate[4]:int(cvsc.shape[1]-coordinate[4])]

    return cvsc


if __name__ == '__main__':
    tnl = read_valid('../align/collection_0000063578_2020-10-13 22_19_26.scn', 'H&E')
    tnl = binarize(tnl)

    itnl = read_valid('../align/collection_0000063576_2020-10-14 14_13_37.scn', 'IHC')
    itnl = binarize(itnl)

    coor, gmax, cvs, he_cvs = optimize(tnl, itnl, 90)

    ovl = overlap(cvs, he_cvs, coor)
    ovl = cvs_to_img(ovl)
    ovl.save('../align/overlap.jpg')
