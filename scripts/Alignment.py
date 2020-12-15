# IHC and HE alignment
from openslide import OpenSlide
import cv2
import numpy as np
from PIL import Image

slide = OpenSlide('../align/collection_0000063578_2020-10-13 22_19_26.scn')
print(slide.level_dimensions)
upperleft = [int(slide.properties['openslide.bounds-x']),
             int(slide.properties['openslide.bounds-y'])]
lowerright = [int(int(slide.properties['openslide.bounds-width'])/4),
              int(int(slide.properties['openslide.bounds-height'])/4)]
x = int(slide.properties['openslide.bounds-width'])-int(slide.properties['openslide.bounds-x'])
y = int(slide.properties['openslide.bounds-height'])-int(slide.properties['openslide.bounds-y'])

print([x, y])

tnl = slide.read_region(upperleft, 1, lowerright).convert('LA')
tnl.save('../align/ori.png')

ihc = OpenSlide('../align/collection_0000063573_2020-10-14 14_11_00.scn')
print(ihc.level_dimensions)
upperleft = [int(ihc.properties['openslide.bounds-x']),
             int(ihc.properties['openslide.bounds-y'])]
lowerright = [int(int(ihc.properties['openslide.bounds-width'])/4),
              int(int(ihc.properties['openslide.bounds-height'])/4)]
x = int(ihc.properties['openslide.bounds-width'])-int(ihc.properties['openslide.bounds-x'])
y = int(ihc.properties['openslide.bounds-height'])-int(ihc.properties['openslide.bounds-y'])

print([x, y])

itnl = ihc.read_region(upperleft, 1, lowerright).convert('LA')
itnl.save('../align/ihc.png')

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


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
    cv2.imwrite("matches.png", imMatches)

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
refFilename = "../align/ori.png"
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Read image to be aligned
imFilename = "../align/ihc.png"
print("Reading image to align : ", imFilename)
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

print("Aligning images ...")
# Registered image will be resotred in imReg.
# The estimated homography will be stored in h.
imReg, h = alignImages(im, imReference)

# Write aligned image to disk.
outFilename = "../align/aligned.png"
print("Saving aligned image : ", outFilename);
cv2.imwrite(outFilename, imReg)

# Print estimated homography
print("Estimated homography : \n", h)

