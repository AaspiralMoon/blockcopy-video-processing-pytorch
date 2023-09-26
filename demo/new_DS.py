import numpy as np

def _costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))

def _checkBounded(xval, yval, w, h, mbSize):
    if ((yval < 0) or
       (yval + mbSize >= h) or
       (xval < 0) or
       (xval + mbSize >= w)):
        return False
    else:
        return True

def new_DS(imgP, imgI, mbSize, p):
    # new Diamond Search method in BlockCache
    #
    # Input
    #   imgP : The image for which we want to find motion vectors
    #   imgI : The reference image
    #   mbSize : Size of the macroblock
    #   p : Search parameter  (read literature to find what this means)
    #
    # Ouput
    #   motionVect : the motion vectors for each integral macroblock in imgP
    #   DScomputations: The average number of points searched for a macroblock

    h, w = imgP.shape

    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((9))*65537

    L = np.floor(np.log2(p + 1))   # ?????

    LDSP = []
    LDSP.append([0, -2])
    LDSP.append([-1, -1])
    LDSP.append([1, -1])
    LDSP.append([-2, 0])
    LDSP.append([0, 0])
    LDSP.append([2, 0])
    LDSP.append([-1, 1])
    LDSP.append([1, 1])
    LDSP.append([0, 2])

    SDSP = []
    SDSP.append([0, -1])
    SDSP.append([-1, 0])
    SDSP.append([0, 0])
    SDSP.append([1, 0])
    SDSP.append([0, 1])

    computations = 0

    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[4] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])           # check large diamond center
            cost = 0
            point = 4
            if costs[4] != 0:
                computations += 1
                for k in range(9):
                    refBlkVer = y + LDSP[k][1]
                    refBlkHor = x + LDSP[k][0]
                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    if k == 4:
                        continue
                    costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations += 1

                point = np.argmin(costs)
                cost = costs[point]

            SDSPFlag = 1            # SDSPFlag = 1, trigger SDSP
            if point != 4:                
                SDSPFlag = 0
                cornerFlag = 1      # cornerFlag = 1: the MBD point is at the corner
                if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):  # check if the MBD point is at the edge
                    cornerFlag = 0
                xLast = x
                yLast = y
                x = x + LDSP[point][0]
                y = y + LDSP[point][1]
                costs[:] = 65537
                costs[4] = cost

            while SDSPFlag == 0:       # start iteration until the SDSP is triggered
                if cornerFlag == 1:    # next MBD point is at the corner
                    for k in range(9):
                        refBlkVer = y + LDSP[k][1]
                        refBlkHor = x + LDSP[k][0]
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        if k == 4:
                            continue

                        if ((refBlkHor >= xLast - 1) and   # avoid redundant computations from the last search
                           (refBlkHor <= xLast + 1) and
                           (refBlkVer >= yLast - 1) and
                           (refBlkVer <= yLast + 1)):
                            continue
                        elif ((refBlkHor < j-p) or         # check if the search goes beyond the search region
                              (refBlkHor > j+p) or
                              (refBlkVer < i-p) or
                              (refBlkVer > i+p)):
                            continue
                        else:
                            costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1
                else:                                # next MBD point is at the edge
                    lst = []
                    if point == 1:                   # the point positions that needs computation
                        lst = np.array([0, 1, 3])
                    elif point == 2:
                        lst = np.array([0, 2, 5])
                    elif point == 6:
                        lst = np.array([3, 6, 8])
                    elif point == 7:
                        lst = np.array([5, 7, 8])

                    for idx in lst:
                        refBlkVer = y + LDSP[idx][1]
                        refBlkHor = x + LDSP[idx][0]
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        elif ((refBlkHor < j - p) or
                              (refBlkHor > j + p) or
                              (refBlkVer < i - p) or
                              (refBlkVer > i + p)):
                            continue
                        else:
                            costs[idx] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1

                point = np.argmin(costs)
                cost = costs[point]

                SDSPFlag = 1
                if point != 4:
                    SDSPFlag = 0
                    cornerFlag = 1
                    if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                        cornerFlag = 0
                    xLast = x
                    yLast = y
                    x += LDSP[point][0]
                    y += LDSP[point][1]
                    costs[:] = 65537
                    costs[4] = cost
            costs[:] = 65537
            costs[2] = cost

            for k in range(5):                # trigger SDSP
                refBlkVer = y + SDSP[k][1]
                refBlkHor = x + SDSP[k][0]

                if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                    continue
                elif ((refBlkHor < j - p) or
                      (refBlkHor > j + p) or
                      (refBlkVer < i - p) or
                      (refBlkVer > i + p)):
                    continue

                if k == 2:
                    continue

                costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                computations += 1

            point = 2
            cost = 0 
            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]

            x += SDSP[point][0]
            y += SDSP[point][1]

            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [x - j, y - i]

            costs[:] = 65537

    return vectors, computations / ((h * w) / mbSize**2)