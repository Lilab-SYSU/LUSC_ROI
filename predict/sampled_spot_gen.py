import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask npy file")
parser.add_argument("grid_path", default=None, metavar="GRID_PATH", type=str,
                    help="Path to the grid point npy file")
parser.add_argument("--patch_size", default=224, metavar="PATCH_SIZE", type=int,
                    help="Tile size")
# parser.add_argument("--patch_number", default=None, metavar="PATCH_NUMB", type=int,
#                     help="The number of patches extracted from WSI",required=False)
parser.add_argument("--level", default=6, metavar="LEVEL", type=float,
                    help="Bool format, whether or not")


class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path, grid_path, patchSize,level,number=2000000):
        self.mask_path = mask_path
        self.grid_path = grid_path
        self.patch_size = patchSize
        self.level = level
        self.number = number

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)
        X_idcs, Y_idcs = np.where(mask_tissue)
        X_idcs = np.rint((X_idcs + 0.5) * self.level)
        Y_idcs = np.rint((Y_idcs + 0.5) * self.level)
        X_idcs = X_idcs.astype(int)
        Y_idcs = Y_idcs.astype(int)

        x = X_idcs - self.patch_size / 2
        y = Y_idcs - self.patch_size / 2
        x = x.astype(int)
        y = y.astype(int)

        grid_points = np.stack(np.vstack((x.T, y.T)), axis=1)

        if grid_points.shape[0] > self.number:
            grid_points = grid_points[np.random.randint(grid_points.shape[0],size=self.number),:]
        else:
            grid_points = grid_points

        np.save(self.grid_path,grid_points)
        return grid_points


def run(args):
    grid_points = patch_point_in_mask_gen(args.mask_path, args.grid_path,args.patch_size,args.level).get_patch_point()
    # sampled_points = (sampled_points * 2 ** args.level).astype(np.int32) # make sure the factor

    # mask_name = os.path.split(args.mask_path)[-1].split(".")[0]
    # name = np.full((sampled_points.shape[0], 1), mask_name)
    # center_points = np.hstack((name, sampled_points))

    grid_path = args.grid_path
    np.save(grid_path, grid_points)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
