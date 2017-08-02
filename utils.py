"""
module of utilities
"""
import cPickle as cp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

def extract_single_stroke_chars_and_digits(data):
    res_data = defaultdict(list)
    for dict_key in data.keys():
        if (ord(dict_key[0]) >= ord('a') and ord(dict_key[0]) <= ord('z'))  or \
            (ord(dict_key[0]) >= ord('A') and ord(dict_key[0]) <= ord('Z')) or \
            (ord(dict_key[0]) >= ord('0') and ord(dict_key[0]) <=ord('9')):
            for d in data[dict_key]:
                #check if this is single stroke char
                if len(d) == 1:
                    res_data[dict_key].append(d[0])
    return res_data

def plot_single_stroke_char_or_digit(data):
    #data is a single stroke with the last entry as the time scale...
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    data_len = len(data[:-1])/2
    ax.plot(data[:data_len], -data[data_len:-1], 'k', linewidth=12.0)
    #<hyin/Feb-9th-2016> we need to carefully define the limits of axes
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax

import Image
# import ImageOps
import os.path
import time
from collections import defaultdict

def generate_images_for_chars_and_digits(data, overwrite=False, grayscale=True, thumbnail_size=(28, 28)):
    img_data = defaultdict(list)

    func_path = os.path.dirname(os.path.realpath(__file__))
    folder = 'bin/images'
    gs_folder = 'bin/grayscale'

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

        for d_idx, d in enumerate(data[dict_key]):
            # print 'Generating images for the {0}-th demonstrations...'.format(d_idx)
            tmp_fname = 'ascii_{0}_{1:04d}.png'.format(ord(dict_key), d_idx)
            tmp_fpath = os.path.join(output_path_char, tmp_fname)
            fig = None
            if not os.path.exists(tmp_fpath) or overwrite:
                print tmp_fpath
                fig, ax = plot_single_stroke_char_or_digit(d)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(tmp_fpath, bbox_inches=extent, dpi=100)

            if grayscale:
                #load the image to have a grayscale file
                # image=Image.open(tmp_fpath).convert("L")
                # inverted_image = ImageOps.invert(image)
                # image.thumbnail(thumbnail_size)
                # arr=np.asarray(image)
                # plt.figimage(arr,cmap=cm.Greys_r)

                tmp_fname_grayscale = 'ascii_{0}_{1:04d}_grayscale_thumbnail.png'.format(ord(dict_key), d_idx)
                tmp_fpath_grayscale = os.path.join(gs_output_path_char, tmp_fname_grayscale)
                if not os.path.exists(tmp_fpath_grayscale) or overwrite:
                    thumbnail = get_char_img_thumbnail(tmp_fpath, tmp_fpath_grayscale)
                    print 'Generating grayscale image {0}'.format(tmp_fname_grayscale)
                else:
                    image = Image.open(tmp_fpath_grayscale)
                    thumbnail = np.asarray(image)
                    # image.close()
                # inverted_image.save(tmp_fpath_grayscale)
                # plt.close(fig)

                #get the np array data for this image
                img_data[char_folder].append(np.asarray(thumbnail))
            if fig is not None:
                plt.close(fig)
            # time.sleep(0.5)
    return img_data

#utilities for computing convenience
from scipy import interpolate

def expand_traj_dim_with_derivative(data, dt=0.01):
    augmented_trajs = []
    for traj in data:
        time_len = len(traj)
        t = np.linspace(0, time_len*dt, time_len)
        if time_len > 3:
            if len(traj.shape) == 1:
                """
                mono-dimension trajectory, row as the entire trajectory...
                """
                spl = interpolate.splrep(t, traj)
                traj_der = interpolate.splev(t, spl, der=1)
                tmp_augmented_traj = np.array([traj, traj_der]).T
            else:
                """
                multi-dimensional trajectory, row as the state variable...
                """
                tmp_traj_der = []
                for traj_dof in traj.T:
                    spl_dof = interpolate.splrep(t, traj_dof)
                    traj_dof_der = interpolate.splev(t, spl_dof, der=1)
                    tmp_traj_der.append(traj_dof_der)
                tmp_augmented_traj = np.vstack([traj.T, np.array(tmp_traj_der)]).T

            augmented_trajs.append(tmp_augmented_traj)

    return augmented_trajs

import dataset as ds

def extract_image_sequences(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given name
        data_dict = cp.load(open(fname, 'rb'))
    def extract_image_helper(d):
        #flatten the image and scale them
        return d.flatten().astype(dtype) * 1./255.

    image_sequences = []
    if data_dict is not None:
        for char in sorted(data_dict.keys(), key=lambda k:k[-1]):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                # if char[-1] == 'o' or char[-1] == 'O':
                image_sequences += [[extract_image_helper(d) for d in s] for s in data_dict[char]]
                    # tmp_s_lst = []
                    # for i in range(len(data_dict[char])):
                    #     tmp_s = []
                    #     for j in range(len(data_dict[char][i])):
                    #         if j == 0:
                    #             tmp_s.append(np.zeros(data_dict[char][i][j].shape))
                    #         else:
                    #             tmp_s.append(data_dict[char][i][j] - data_dict[char][i][j-1])
                    #     tmp_s_lst.append(tmp_s)
                    # # image_sequences += [[extract_image_helper(d) for d in s] for s in data_dict[char]]
                    # image_sequences += [[extract_image_helper(d) for d in s] for s in tmp_s_lst]
    return np.array(image_sequences)


def extract_images(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given name
        data_dict = cp.load(open(fname, 'rb'))

    def extract_image_helper(d):
        #flatten the image and scale them
        return d.flatten().astype(dtype) * 1./255.
    images = []
    if data_dict is not None:
        for char in sorted(data_dict.keys(), key=lambda k:k[-1]):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                images += [extract_image_helper(d) for d in data_dict[char]]
    return np.array(images)

def extract_jnt_trajs(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_jnt_trajs_helper(d):
        #flatten the image and scale them, is it necessary for joint trajectory, say within pi radians?
        return d.flatten().astype(dtype)
    jnt_trajs = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                jnt_trajs += [extract_jnt_trajs_helper(d) for d in data_dict[char]]
    return np.array(jnt_trajs)

def extract_jnt_fa_parms(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    fa_parms = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                fa_parms += [d for d in data_dict[char]]
    fa_parms = np.array(fa_parms)
    #Gaussian statistics for potential normalization
    fa_mean = np.mean(fa_parms, axis=0)
    fa_std = np.std(fa_parms, axis=0)
    return fa_parms, fa_mean, fa_std

import cv2

'''
utility to threshold the character image
'''
def threshold_char_image(img):
    #do nothing for now
    return img
'''
utility to segment character contour and get the rectangular bounding box
'''
def segment_char_contour_bounding_box(img):
    cv2_version = cv2.__version__.split('.')
    # print cv2_version
    if int(cv2_version[0]) < 3:
        # for opencv below 3.0.0
        ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # for opencv from 3.0.0
        _, ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    #for blank image
    if len(rects) == 0:
        return [0, 0, img.shape[1], img.shape[0]]
    #rect length-4 array, (rect[0], rect[1]) - lower left corner point, (rect[2], rect[3]) - width, height
    corner_pnts = []
    for rect in rects:
        corner_pnts.append([rect[0], rect[1]])
        corner_pnts.append([rect[0]+rect[2], rect[1]+rect[3]])
    corner_pnts = np.array(corner_pnts)
    l_corner_pnt = np.amin(corner_pnts, axis=0)
    u_corner_pnt = np.amax(corner_pnts, axis=0)
    return [l_corner_pnt[0], l_corner_pnt[1], u_corner_pnt[0]-l_corner_pnt[0], u_corner_pnt[1]-l_corner_pnt[1]]
'''
utility to resize
'''
def get_char_img_thumbnail_helper(img_data):
    #first threshold the img
    thres_img = threshold_char_image(img_data)
    #then figure out the contour bouding box
    bound_rect = segment_char_contour_bounding_box(thres_img)
    center = [bound_rect[0] + bound_rect[2]/2., bound_rect[1] + bound_rect[3]/2.]
    #crop the interested part
    leng = max([int(bound_rect[2]), int(bound_rect[3])])
    border = int(0.6*leng)
    pt1 = int(center[1] -bound_rect[3] // 2)
    pt2 = int(center[0] -bound_rect[2] // 2)
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
    roi = cv2.resize(cv_img_bckgrnd, (56, 56), interpolation=cv2.INTER_AREA)
    return roi, bound_rect
def get_char_img_thumbnail(img_fname, gs_fname):
    #convert this pil image to the cv one
    cv_img = cv2.imread(img_fname)
    cv_img_gs = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2GRAY)
    cv_img_gs_inv = 255 - cv_img_gs
    roi, _ = get_char_img_thumbnail_helper(cv_img_gs_inv)
    # roi = cv2.dilate(roi, (3, 3))
    #write this image
    cv2.imwrite(gs_fname, roi)
    return roi

def display_image_sequences(image_seq, img_size=28):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for img in image_seq:
        ax.imshow(np.reshape(img, (img_size, -1)), interpolation='nearest')
        plt.draw()
        raw_input()
    return

import matplotlib
matplotlib.use('Agg')
import time
import os
from matplotlib import animation

def generate_image_sequence_files(image_seq, img_size=28, folder='figs', anim=True):
    #generate figures for them as well as gifs...
    # fig = plt.figure(frameon = False, dpi=100)
    # plt.ion()
    fig = plt.figure(frameon=False, figsize=(4,4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.add_axes(ax)
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fname = time.strftime("%Y%m%d%H%M%S")

    img_obj = ax.imshow(np.zeros((img_size, img_size)), 'gray', interpolation='bilinear', vmax=1.0, vmin=0.0)
    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    def init():
        img = image_seq[0]
        img_obj.set_data(np.reshape(img, (img_size, -1)))
        return (img_obj, )

    def animate(i):
        img = image_seq[i]
        img_obj.set_data(np.reshape(img, (img_size, -1)))
        return (img_obj, )

    for i, img in enumerate(image_seq):
        # img_obj = ax.imshow(np.reshape(img, (img_size, -1)), 'gray', interpolation='bilinear')
        # ax.set_aspect('equal')
        # plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        img_obj.set_data(np.reshape(img, (img_size, -1)))
        fig.canvas.draw()
        tmp_fname = os.path.join(folder, '{}_Step{:04d}'.format(fname, i))
        fig.savefig(tmp_fname, bbox_inches=extent, dpi=100)
        # with open(tmp_fname, 'w') as outfile:
        #     fig.canvas.print_png(outfile)
        # plt.savefig(tmp_fname, bbox_inches='tight', dpi=100)

    if anim:
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20, interval=20, blit=True)
        tmp_gifname = os.path.join(folder, '{}_{}.gif'.format(fname, 'anim'))
        anim.save(tmp_gifname, writer='imagemagick', fps=5)
        plt.show()
    return
