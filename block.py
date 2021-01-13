import numpy as np
from skvideo.utils import vshape, rgb2gray
from numba import jit

@jit(nopython=True)
def _N3SS(imgP, imgI, mbSize, p):
    # Computes motion vectors using *NEW* Three Step Search method
    #
    # Input
    #   imgP : The image for which we want to find motion vectors
    #   imgI : The reference image
    #   mbSize : Size of the macroblock
    #   p : Search parameter  (read literature to find what this means)
    #
    # Ouput
    #   motionVect : the motion vectors for each integral macroblock in imgP
    #   NTSScomputations: The average number of points searched for a macroblock

    h, w = imgP.shape

    h = np.int(h/mbSize) * mbSize
    w = np.int(w/mbSize) * mbSize
    imgP = imgP[:h, :w]
    imgI = imgI[:h, :w]

    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2), dtype=np.float32)

    costs = np.ones((3, 3), dtype=np.float32)*65537

    computations = 0

    L = np.floor(np.log2(p + 1))
    stepMax = np.int(2**(L - 1))

    l_count = 0
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            
            costs[1, 1] = np.mean(np.abs(imgP[i:i + mbSize, j:j + mbSize].astype(np.float32) - \
                 imgI[i:i + mbSize, j:j + mbSize].astype(np.float32)))
            computations += 1

            stepSize = stepMax

            for m in range(-stepSize, stepSize + 1, stepSize):
                for n in range(-stepSize, stepSize + 1, stepSize):
                    refBlkVer = y + m
                    refBlkHor = x + n
                    if ((refBlkVer < 0) or
                       (refBlkVer + mbSize > h) or
                       (refBlkHor < 0) or
                       (refBlkHor + mbSize > w)):
                            continue
                    costRow = np.int(m / stepSize) + 1
                    costCol = np.int(n / stepSize) + 1
                    if ((costRow == 1) and (costCol == 1)):
                        continue
                    costs[costRow, costCol] = np.mean(np.abs(imgP[i:i + mbSize, j:j + mbSize].astype(np.float32) - \
                 imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize].astype(np.float32)))
                    computations = computations + 1

            
            min1 = costs.min()
            dy,dx = divmod(costs.argmin(), costs.shape[1])
            if min1 == costs[1,1]:
                dy,dx = 1,1

            x1 = x + (dx - 1) * stepSize
            y1 = y + (dy - 1) * stepSize

            stepSize = 1
            for m in range(-stepSize, stepSize + 1, stepSize):
                for n in range(-stepSize, stepSize + 1, stepSize):
                    refBlkVer = y + m
                    refBlkHor = x + n
                    if ((refBlkVer < 0) or
                       (refBlkVer + mbSize > h) or
                       (refBlkHor < 0) or
                       (refBlkHor + mbSize > w)):
                            continue
                    costRow = m + 1
                    costCol = n + 1
                    if ((costRow == 1) and (costCol == 1)):
                        continue
                    costs[costRow, costCol] = np.mean(np.abs(imgP[i:i + mbSize, j:j + mbSize].astype(np.float32) - \
                 imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize].astype(np.float32)))
                    computations += 1
            
            min2 = costs.min()
            dy,dx = divmod(costs.argmin(), costs.shape[1])
            if min2 == costs[1,1]:
                dy,dx = 1,1
            x2 = x + (dx - 1)
            y2 = y + (dy - 1)

            NTSSFlag = 0
            if ((x1 == x2) and (y1 == y2)):
                NTSSFlag = -1
                #x = x1
                #y = y1
            elif (min2 <= min1):
                x = x2
                y = y2
                NTSSFlag = 1
            else:
                x = x1
                y = y1

            if NTSSFlag == 1:
                costs[:, :] = 65537
                costs[1, 1] = min2
                stepSize = 1
                for m in range(-stepSize, stepSize + 1, stepSize):
                    for n in range(-stepSize, stepSize + 1, stepSize):
                        refBlkVer = y + m
                        refBlkHor = x + n
                        if ((refBlkVer < 0) or
                           (refBlkVer + mbSize > h) or
                           (refBlkHor < 0) or
                           (refBlkHor + mbSize > w)):
                                continue

                        if ((refBlkVer >= i - 1) and
                            (refBlkVer <= i + 1) and
                            (refBlkHor >= j - 1) and
                            (refBlkHor <= j + 1)):
                                continue
                        costRow = np.int(m/stepSize) + 1
                        costCol = np.int(n/stepSize) + 1
                        if ((costRow == 1) and (costCol == 1)):
                            continue
                        costs[costRow, costCol] = np.mean(np.abs(imgP[i:i + mbSize, j:j + mbSize].astype(np.float32) - \
                 imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize].astype(np.float32)))
                        computations += 1
                
                min2 = costs.min()
                dy,dx = divmod(costs.argmin(), costs.shape[1])
                if min2 == costs[1,1]:
                    dy,dx = 1,1

                x += (dx - 1)
                y += (dy - 1)


            elif NTSSFlag == 0:
                costs[:, :] = 65537
                costs[1, 1] = min1
                stepSize = np.int(stepMax / 2)
                while(stepSize >= 1):
                    for m in range(-stepSize, stepSize+1, stepSize):
                        for n in range(-stepSize, stepSize+1, stepSize):
                            refBlkVer = y + m
                            refBlkHor = x + n
                            if ((refBlkVer < 0) or
                               (refBlkVer + mbSize > h) or
                               (refBlkHor < 0) or
                               (refBlkHor + mbSize > w)):
                                    continue
                            costRow = np.int(m / stepSize) + 1
                            costCol = np.int(n / stepSize) + 1
                            if ((costRow == 1) and (costCol == 1)):
                                continue
                            costs[costRow, costCol] = np.mean(np.abs(imgP[i:i + mbSize, j:j + mbSize].astype(np.float32) - \
                 imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize].astype(np.float32)))
                            
                            computations = computations + 1
                            l_count += 1
                    mi = costs.min()
                    dy,dx = divmod(costs.argmin(), costs.shape[1])
                    if mi == costs[1,1]:
                        dy,dx = 1,1

                    x += (dx - 1) * stepSize
                    y += (dy - 1) * stepSize

                    stepSize = np.int(stepSize / 2)
                    costs[1, 1] = costs[dy, dx]


            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [y - i, x - j]

            costs[:, :] = 65537

    return vectors, computations / ((h * w) / mbSize**2)

def blockMotion(videodata, method='DS', mbSize=8, p=2, **plugin_args):
    videodata = vshape(videodata)

    # grayscale
    luminancedata = rgb2gray(videodata)

    numFrames, height, width, channels = luminancedata.shape
    assert numFrames > 1, "Must have more than 1 frame for motion estimation!"

    # luminance is 1 channel, so flatten for computation
    luminancedata = luminancedata.reshape((numFrames, height, width))

    motionData = np.zeros((numFrames - 1, np.int(height / mbSize), np.int(width / mbSize), 2), np.int8)

    if method == "N3SS":
        for i in range(numFrames - 1):
            motion, comps = _N3SS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    else:
        raise NotImplementedError

    return motionData