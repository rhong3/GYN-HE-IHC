# IHC and HE alignment
from openslide import OpenSlide
import cv2
import numpy as np
from PIL import Image
import staintools

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


def white(img):
    img = np.array(img)[:, :, :3]
    img = np.nan_to_num(img, nan=255, posinf=255, neginf=255)
    mask = ((img[:, :, :3] > 200).astype(np.uint8) + (img[:, :, :3] < 50).astype(np.uint8))*255
    img = img + mask
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


slide = OpenSlide('../align/collection_0000063578_2020-10-13 22_19_26.scn')
print(slide.level_dimensions)
upperleft = [int(slide.properties['openslide.bounds-x']),
             int(slide.properties['openslide.bounds-y'])]
lowerright = [int(int(slide.properties['openslide.bounds-width'])/4),
              int(int(slide.properties['openslide.bounds-height'])/4)]
x = int(slide.properties['openslide.bounds-width'])-int(slide.properties['openslide.bounds-x'])
y = int(slide.properties['openslide.bounds-height'])-int(slide.properties['openslide.bounds-y'])

print([x, y])

tnl = slide.read_region(upperleft, 1, lowerright).convert('RGB')

tnl = normalization(tnl, std)
tnl = white(tnl)

tnl.save('../align/ori.jpg')

# tnl=cv2.imread('../align/ori.jpg', 0)
# tnl = cv2.adaptiveThreshold(tnl,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# cv2.imwrite('../align/ori-x.jpg', tnl)


ihc = OpenSlide('../align/collection_0000063573_2020-10-14 14_11_00.scn')
print(ihc.level_dimensions)
upperleft = [int(ihc.properties['openslide.bounds-x']),
             int(ihc.properties['openslide.bounds-y'])]
lowerright = [int(int(ihc.properties['openslide.bounds-width'])/4),
              int(int(ihc.properties['openslide.bounds-height'])/4)]
x = int(ihc.properties['openslide.bounds-width'])-int(ihc.properties['openslide.bounds-x'])
y = int(ihc.properties['openslide.bounds-height'])-int(ihc.properties['openslide.bounds-y'])

print([x, y])

itnl = ihc.read_region(upperleft, 1, lowerright).convert('RGB')

itnl = white(itnl)

itnl.save('../align/ihc.jpg')

# itnl = cv2.imread('../align/ihc.jpg', 0)
# itnl = cv2.adaptiveThreshold(itnl,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# cv2.imwrite('../align/ihc-x.jpg', itnl)


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.5


def alignImages(im1, im2):
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


# Read reference image
refFilename = "../align/ori.jpg"
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Read image to be aligned
imFilename = "../align/ihc.jpg"
print("Reading image to align : ", imFilename)
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

print("Aligning images ...")
# Registered image will be resotred in imReg.
# The estimated homography will be stored in h.
imReg, h = alignImages(im, imReference)

# Write aligned image to disk.
outFilename = "../align/aligned.jpg"
print("Saving aligned image : ", outFilename)
cv2.imwrite(outFilename, imReg)

# Print estimated homography
print("Estimated homography : \n", h)

