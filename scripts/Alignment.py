# IHC and HE alignment
from openslide import OpenSlide
import cv2
import numpy as np
from PIL import Image
import staintools
import skimage.morphology as skm
from scipy.ndimage.interpolation import rotate

std = staintools.read_image("../colorstandard.png")
std = staintools.LuminosityStandardizer.standardize(std)


# Tile color normalization
def normalization(img, sttd):
    img = np.array(img)[:, :, :3]
    img = staintools.LuminosityStandardizer.standardize(img)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(sttd)
    img = normalizer.transform(img)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


def binarize(img, name):
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
    mask = skm.remove_small_objects(mask, min_size=100000, connectivity=1, in_place=False)
    mask = skm.remove_small_holes(mask, area_threshold=100000, connectivity=1, in_place=False)

    mask_out = mask*255
    mask_out = Image.fromarray(mask_out.astype('uint8'), 'RGB')
    mask_out.save(name)

    return mask


def alignImages(im1, im2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.5
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("../align/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def optimize(imga, imgb):
    imgb = imgb[:, :, 0]
    pdd = int(np.sqrt(imgb.shape[0]**2+imgb.shape[1]**2)-np.amin(imga.shape[:2])+1)
    imga = imga[:, :, 0]
    imga = np.pad(imga, pdd, mode=pad_with).astype('uint8')

    ori_pad_out = imga*255
    ori_pad_out_RGB = np.empty([ori_pad_out.shape[0], ori_pad_out.shape[1], 3])
    ori_pad_out_RGB[:, :, 0] = ori_pad_out
    ori_pad_out_RGB[:, :, 1] = ori_pad_out
    ori_pad_out_RGB[:, :, 2] = ori_pad_out
    ori_pad_out_RGB = Image.fromarray(ori_pad_out_RGB.astype('uint8'), 'RGB')
    ori_pad_out_RGB.save('../align/ori_pad.jpg')

    ori_canvas = np.zeros(imga.shape)
    ori_canvas[pdd:pdd+imgb.shape[0], pdd:pdd+imgb.shape[1]] = imgb

    ori_canvas_out = ori_canvas*255
    ori_canvas_out_RGB = np.empty([ori_canvas_out.shape[0], ori_canvas_out.shape[1], 3])
    ori_canvas_out_RGB[:, :, 0] = ori_canvas_out
    ori_canvas_out_RGB[:, :, 1] = ori_canvas_out
    ori_canvas_out_RGB[:, :, 2] = ori_canvas_out
    ori_canvas_out_RGB = Image.fromarray(ori_canvas_out_RGB.astype('uint8'), 'RGB')
    ori_canvas_out_RGB.save('../align/ori_canvas.jpg')

    globalmax = 0
    for t in range(2):
        if t == 0:
            imgbt = imgb
        else:
            imgbt = np.transpose(imgb)
        for r in range(24):
            imgx = rotate(imgbt, r*15)
            istart = 0
            jstart = 0
            iend = imga.shape[0]-imgx.shape[0]
            jend = imga.shape[1]-imgx.shape[1]
            step = int(np.amax([iend, jend]) / 5)
            maxx = 0
            best_coor = [t, r, istart, jstart, pdd]
            sndbest_coor = [t, r, iend, jend, pdd]
            while step >= 10:
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
                            print("new local max: {}".format(maxx))
                            print('new local best coordinate: ')
                            print(best_coor)
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
                    step = int(step/10)
            if maxx > globalmax:
                globalmax = maxx
                globalmax_coor = best_coor
                globalbest_canvas = best_canvas
                print("new global max: {}".format(globalmax))
                print('new global best coordinate: ')
                print(globalmax_coor)

    print("Global max: {}".format(globalmax))
    print('Global best coordinate: ')
    print(globalmax_coor)

    best_canvas_out = globalbest_canvas*255
    best_canvas_out_RGB = np.empty([best_canvas_out.shape[0], best_canvas_out.shape[1], 3])
    best_canvas_out_RGB[:, :, 0] = best_canvas_out
    best_canvas_out_RGB[:, :, 1] = best_canvas_out
    best_canvas_out_RGB[:, :, 2] = best_canvas_out
    best_canvas_out_RGB = Image.fromarray(best_canvas_out_RGB.astype('uint8'), 'RGB')
    best_canvas_out_RGB.save('../align/best_canvas.jpg')


slide = OpenSlide('../align/collection_0000063578_2020-10-13 22_19_26.scn')
print(slide.level_dimensions)
upperleft = [int(slide.properties['openslide.bounds-x']),
             int(slide.properties['openslide.bounds-y'])]
lowerright = [int(int(slide.properties['openslide.bounds-width'])/16),
              int(int(slide.properties['openslide.bounds-height'])/16)]
x = int(slide.properties['openslide.bounds-width'])-int(slide.properties['openslide.bounds-x'])
y = int(slide.properties['openslide.bounds-height'])-int(slide.properties['openslide.bounds-y'])

print([x, y])

tnl = slide.read_region(upperleft, 2, lowerright).convert('RGB')
tnl = binarize(tnl, '../align/ori.jpg')


ihc = OpenSlide('../align/collection_0000063573_2020-10-14 14_11_00.scn')
print(ihc.level_dimensions)
upperleft = [int(ihc.properties['openslide.bounds-x']),
             int(ihc.properties['openslide.bounds-y'])]
lowerright = [int(int(ihc.properties['openslide.bounds-width'])/16),
              int(int(ihc.properties['openslide.bounds-height'])/16)]
x = int(ihc.properties['openslide.bounds-width'])-int(ihc.properties['openslide.bounds-x'])
y = int(ihc.properties['openslide.bounds-height'])-int(ihc.properties['openslide.bounds-y'])

print([x, y])

itnl = ihc.read_region(upperleft, 2, lowerright).convert('RGB')
itnl = binarize(itnl, '../align/ihc.jpg')

optimize(tnl, itnl)


