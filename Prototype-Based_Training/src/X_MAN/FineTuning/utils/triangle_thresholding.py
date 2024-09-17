import numpy as np

class ComputeTriangleThreshold:
    def __init__(self):
        pass

    @staticmethod
    def compute_mask(img : np.ndarray):
        histogram = np.histogram(img, bins=256, range=(0, 256))[0]
        threshold = ComputeTriangleThreshold.compute_bin(histogram)
        mask = img > threshold
        return mask
    
    @staticmethod
    def compute_bin(histogram):
        
        # Find min and max bins
        min_bin = np.nonzero(histogram)[0][0]
        max_bin = np.argmax(histogram)
        min_bin2 = np.nonzero(histogram)[0][-1]
        
        if min_bin > 0:
            min_bin -= 1  # Line to the (p==0) point, not to histogram[min]

        if min_bin2 < len(histogram) - 1:
            min_bin2 += 1

        # Determine if histogram should be inverted
        inverted = False
        if (max_bin - min_bin) < (min_bin2 - max_bin):
            inverted = True
            histogram = histogram[::-1]
            min_bin = len(histogram) - 1 - min_bin2
            max_bin = len(histogram) - 1 - max_bin

        if min_bin == max_bin:
            return min_bin

        # Describe line by nx * x + ny * y - d = 0
        nx = histogram[max_bin]
        ny = min_bin - max_bin
        d = np.sqrt(nx * nx + ny * ny)
        nx /= d
        ny /= d
        d = nx * min_bin + ny * histogram[min_bin]

        # Find split point
        split = min_bin
        split_distance = 0
        for i in range(min_bin + 1, max_bin + 1):
            new_distance = nx * i + ny * histogram[i] - d
            if new_distance > split_distance:
                split = i
                split_distance = new_distance

        split -= 1

        # Reverse the histogram back if it was inverted
        if inverted:
            histogram = histogram[::-1]
            return len(histogram) - 1 - split

        return split