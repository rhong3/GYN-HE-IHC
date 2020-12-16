# IHC and HE alignment
from openslide import OpenSlide
import cv2
import numpy as np
from PIL import Image
import staintools
import skimage.morphology as skm

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
    pdd = int(np.amax(imgb.shape)-np.amin(imga.shape)+1)
    print(pdd)
    imga = imga[:, :, 0]
    imga = np.pad(imga, pdd, mode=pad_with).astype('uint8')

    ori_pad_out = imga*255
    ori_pad_out_RGB = np.empty([ori_pad_out.shape[0], ori_pad_out.shape[1], 3])
    ori_pad_out_RGB[:, :, 0] = ori_pad_out
    ori_pad_out_RGB[:, :, 1] = ori_pad_out
    ori_pad_out_RGB[:, :, 2] = ori_pad_out
    ssss = Image.fromarray(ori_pad_out_RGB.astype('uint8'), 'RGB')
    ssss.save('../align/ori_pad.jpg')

    ori_canvas = np.zeros(imga.shape)
    ori_canvas[pdd:pdd+imgb.shape[0], pdd:pdd+imgb.shape[1]] = imgb

    ori_canvas_out = ori_canvas*255
    ori_canvas_out_RGB = np.empty([ori_canvas_out.shape[0], ori_canvas_out.shape[1], 3])
    ori_canvas_out_RGB[:, :, 0] = ori_canvas_out
    ori_canvas_out_RGB[:, :, 1] = ori_canvas_out
    ori_canvas_out_RGB[:, :, 2] = ori_canvas_out
    ori_canvas_out_RGB = Image.fromarray(ori_canvas_out_RGB.astype('uint8'), 'RGB')
    ori_canvas_out_RGB.save('../align/ori_canvas.jpg')

    maxx = 0
    for t in range(2):
        if t == 0:
            pass
        else:
            imgb = np.transpose(imgb)
        for r in range(4):
            imgx = np.rot90(imgb, r)
            for i in range(0, imga.shape[0]-imgx.shape[0], 1000):
                for j in range(0, imga.shape[1]-imgx.shape[1], 1000):
                    canvas = np.zeros(imga.shape)
                    canvas[i:i + imgx.shape[0], j:j + imgx.shape[1]] = imgx
                    canvas = canvas.astype('uint8')

                    summ = np.sum(np.multiply(imga, canvas))
                    if summ > maxx:
                        maxx = summ
                        best_canvas = canvas
                        best_coor = [t, r, i, j]
                        print(maxx)
                        print(best_coor)

    best_canvas_out = best_canvas*255
    best_canvas_out_RGB = np.empty([best_canvas_out.shape[0], best_canvas_out.shape[1], 3])
    best_canvas_out_RGB[:, :, 0] = best_canvas_out
    best_canvas_out_RGB[:, :, 1] = best_canvas_out
    best_canvas_out_RGB[:, :, 2] = best_canvas_out
    best_canvas_out_RGB = Image.fromarray(best_canvas_out_RGB.astype('uint8'), 'RGB')
    best_canvas_out_RGB.save('../align/best_canvas.jpg')


    # # Read reference image
    # refFilename = "../align/ori_canvas.jpg"
    # print("Reading reference image : ", refFilename)
    # imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    #
    # # Read image to be aligned
    # imFilename = "../align/best_canvas.jpg"
    # print("Reading image to align : ", imFilename)
    # im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    #
    # print("Aligning images ...")
    # # Registered image will be resotred in imReg.
    # # The estimated homography will be stored in h.
    # imReg, h = alignImages(im, imReference)
    #
    # # Write aligned image to disk.
    # outFilename = "../align/aligned.jpg"
    # print("Saving aligned image : ", outFilename)
    # cv2.imwrite(outFilename, imReg)
    #
    # # Print estimated homography
    # print("Estimated homography : \n", h)
    #
    # return h



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


