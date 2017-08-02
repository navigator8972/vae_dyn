import cPickle as cp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import Image
import cv2
# import ImageOps
import os.path
import time

import argparse
import utils

def get_expected_size(img_fname):
    cv_img = cv2.imread(img_fname)
    cv_img_gs = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2GRAY)
    img_data = 255 - cv_img_gs
    #first threshold the img
    thres_img = utils.threshold_char_image(img_data)
    #then figure out the contour bouding box
    bound_rect = utils.segment_char_contour_bounding_box(thres_img)
    center = [bound_rect[0] + bound_rect[2]/2., bound_rect[1] + bound_rect[3]/2.]
    #crop the interested part
    leng = max([int(bound_rect[2]), int(bound_rect[3])])
    border = int(0.6*leng)
    pt1 = int(center[1] -bound_rect[3] // 2)
    pt2 = int(center[0] -bound_rect[2] // 2)
    return bound_rect, leng, border, pt1, pt2

def get_char_img_thumbnail_helper(img_data, bound_rect, leng, border, pt1, pt2, img_size):
    # <hyin/Aug-15th-2016> change this from one to zero. Note this is different from the current generated data whose background color is not "fully dark"...
    cv_img_bckgrnd = np.zeros((border+leng, border+leng))
    # print cv_img_bckgrnd.shape
    # print bound_rect
    # print center
    # print border
    # print (pt1+border//2),(pt1+bound_rect[3]+border//2), (pt2+border//2),(pt2+bound_rect[2]+border//2)
    # print cv_img_bckgrnd[(border//2):(bound_rect[3]+border//2), (border//2):(bound_rect[2]+border//2)].shape

    cv_img_bckgrnd[ (border//2+(leng-bound_rect[3])//2):(bound_rect[3]+border//2+(leng-bound_rect[3])//2),
                    (border//2+(leng-bound_rect[2])//2):(bound_rect[2]+border//2+(leng-bound_rect[2])//2)] = img_data[pt1:(pt1+bound_rect[3]), pt2:(pt2+bound_rect[2])]
    # roi = cv_img_gs_inv[pt1:(pt1+border*2+leng), pt2:(pt2+border*2+leng)]
    # Resize the image
    roi = cv2.resize(cv_img_bckgrnd, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return roi, bound_rect

def get_char_img_thumbnail(img_fname, gs_fname, bound_rect, leng, border, pt1, pt2, img_size):
    #convert this pil image to the cv one
    cv_img = cv2.imread(img_fname)
    cv_img_gs = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2GRAY)
    cv_img_gs_inv = 255 - cv_img_gs
    roi, _ = get_char_img_thumbnail_helper(cv_img_gs_inv, bound_rect, leng, border, pt1, pt2, img_size)
    # roi = cv2.dilate(roi, (3, 3))
    #write this image
    cv2.imwrite(gs_fname, roi)
    return roi

def main(args):
    """
    """
    img_data = defaultdict(list)
    print 'Loading data pickle file...'
    data = cp.load(open(args.ujichar_dataset, 'rb'))

    func_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.save_dir)
    folder = 'images'
    gs_folder = 'grayscale'

    output_path = os.path.join(func_path, folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gs_output_path = os.path.join(func_path, gs_folder)
    if not os.path.exists(gs_output_path):
        os.makedirs(gs_output_path)

    for dict_key in data.keys():
        print 'Processing character or digit {0}'.format(dict_key)
        char_folder = 'char_{0}_{1}'.format(ord(dict_key), dict_key)
        output_path_char = os.path.join(output_path, char_folder)
        if not os.path.exists(output_path_char):
            os.makedirs(output_path_char)

        gs_output_path_char = os.path.join(gs_output_path, char_folder)
        if not os.path.exists(gs_output_path_char):
            os.makedirs(gs_output_path_char)

        s_indices = np.linspace(0, len(data[dict_key])-1, args.nsample_per_char).astype(int)
        for d_idx, s_idx in enumerate(s_indices):
            d = data[dict_key][s_idx]
            tmp_len = len(d[:-1])/2
            end_indices = np.linspace(0, tmp_len, args.seq_length).astype(int)
            tmp_seq_img = []
            for seq_idx, end_ind in enumerate(end_indices):
                tmp_d = np.concatenate([d[:end_ind], d[tmp_len:tmp_len+end_ind], [d[-1]]])
                fig, ax = utils.plot_single_stroke_char_or_digit(tmp_d)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                tmp_fname = 'ascii_{0}_{1:04d}_seq{2}.png'.format(ord(dict_key), d_idx, seq_idx)
                tmp_fpath = os.path.join(output_path_char, tmp_fname)
                fig.savefig(tmp_fpath, bbox_inches=extent, dpi=100)
                if fig is not None:
                    plt.close(fig)

            if args.grayscale:
                final_img_fname = tmp_fpath
                bound_rect, leng, border, pt1, pt2 = get_expected_size(tmp_fpath)
                for seq_idx, end_ind in enumerate(end_indices):
                    tmp_fname = 'ascii_{0}_{1:04d}_seq{2}.png'.format(ord(dict_key), d_idx, seq_idx)
                    tmp_fpath = os.path.join(output_path_char, tmp_fname)

                    tmp_fname_grayscale = 'ascii_{0}_{1:04d}_seq{2}_grayscale_thumbnail.png'.format(ord(dict_key), d_idx, seq_idx)
                    tmp_fpath_grayscale = os.path.join(gs_output_path_char, tmp_fname_grayscale)

                    if not os.path.exists(tmp_fpath_grayscale) or args.overwrite:
                        thumbnail = get_char_img_thumbnail(tmp_fpath, tmp_fpath_grayscale, bound_rect, leng, border, pt1, pt2, args.img_size)
                        print 'Generating grayscale image {0}'.format(tmp_fname_grayscale)
                    else:
                        image = Image.open(tmp_fpath_grayscale)
                        thumbnail = np.asarray(image)
                    #get the np array data for this image
                    tmp_seq_img.append(np.asarray(thumbnail))


            img_data[char_folder].append(tmp_seq_img)

    img_data_file_path = os.path.join(func_path, 'extracted_data_image_seq.pkl')
    cp.dump(img_data, open(img_data_file_path, 'wb'))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ujichar_dataset', type=str, default='bin/extracted_data_extend.pkl',
                        help='file of ujichar dataset')
    parser.add_argument('--nsample_per_char', type=int, default=32,
                        help='number of samples for each character')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='length of sampled sequence')
    parser.add_argument('--img_size', type=int, default=56,
                        help='size of images')
    parser.add_argument('--save_dir', type=str, default='bin',
                        help='directory to save the data')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing image files')
    parser.add_argument('--grayscale', type=int, default=1,
                        help='use grayscale images')
    args = parser.parse_args()

    main(args)
