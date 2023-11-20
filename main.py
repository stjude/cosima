import os
import sys
import copy
import json
import csv
import datetime
from time import time
import re
import math
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import mpl_interactions.ipyplot as iplt
from PIL import Image
import shutil
from scipy import interpolate


pl.Config.set_fmt_str_lengths(1000)
pl.Config.set_tbl_rows(1000)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=500)

INPUT_FORMAT = {".tif": 0,".csv": 0}

ILASTIK_COLS = ['object_id',
                'Predicted Class',
                'Probability of LipidDroplets',
                'Object Center_0',
                'Object Center_1',
                'Object Area',
                'Radii of the object_0',
                'Radii of the object_1',
                'Size in pixels',
                'Bounding Box Maximum_0',
                'Bounding Box Maximum_1',
                'Bounding Box Minimum_0',
                'Bounding Box Minimum_1',
                'Diameter']

ct = datetime.datetime.now()
timestamp = ct.strftime("D%Y_%m_%dT%H_%M_%S")
print(timestamp)

output_path = os.path.realpath("./output/output_" + timestamp)
os.makedirs(output_path)
err_log = {'impact': [], 'e_msg': [], 'type': [], 'source': []}
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]

WALKER_MAX = 253
X_NORMALIZE = 400
BK_SUB, SHOW_OVERLAP = True, True
RING_THICKNESS = 3
ERODE_THICKNESS = 3
INPUT_PATH = os.path.realpath(r'./input')
RDL_AVER = True
RDL_AVER_START = 0
RDL_AVER_END = RING_THICKNESS-1
SHOW_PLOTS = True
CHANNELS = {0:'halo', 1:'fabccon'}
B_CHANNEL = 0
# if len(sys.argv) == 1:
#     RING_THICKNESS = 5
#     INPUT_PATH = os.path.realpath(r'./input')
#     PG_START = 0
#     PG_END = RING_THICKNESS

SETTING_ITEMS = {
    'X_NORMALIZE': X_NORMALIZE,
    'BK_SUB': BK_SUB,
    'SHOW_OVERLAP': SHOW_OVERLAP,
    'RING_THICKNESS': RING_THICKNESS,
    'ERODE_THICKNESS': ERODE_THICKNESS,
    'INPUT_PATH': INPUT_PATH,
    'RDL_AVER': RDL_AVER,
    'RDL_AVER_START': RDL_AVER_START,
    'RDL_AVER_END': RDL_AVER_END,
    'SHOW_PLOTS': SHOW_PLOTS,
    'CHANNELS': CHANNELS,
    'B_CHANNEL': B_CHANNEL,
}

if len(sys.argv) > 1:
    PARAMETERS = eval(sys.argv[1])

    print(PARAMETERS)
    if PARAMETERS['INPUT_PATH']:
        INPUT_PATH = PARAMETERS['INPUT_PATH']

    if type(PARAMETERS['ERODE_THICKNESS']) is int:
        ERODE_THICKNESS = PARAMETERS['ERODE_THICKNESS']

    if PARAMETERS['RING_THICKNESS']:
        RING_THICKNESS = PARAMETERS['RING_THICKNESS']

    if not PARAMETERS['BK_SUB']:
        print(PARAMETERS['BK_SUB'])
        BK_SUB = PARAMETERS['BK_SUB']

    if not PARAMETERS['SHOW_OVERLAP']:
        print(PARAMETERS['SHOW_OVERLAP'])
        SHOW_OVERLAP = PARAMETERS['SHOW_OVERLAP']

    if type(PARAMETERS['RDL_AVER']) is bool:
        RDL_AVER = PARAMETERS['RDL_AVER']

    if type(PARAMETERS['RDL_AVER_START']) is int:
        RDL_AVER_START = PARAMETERS['RDL_AVER_START']-1

    if type(PARAMETERS['RDL_AVER_END']) is int:
        RDL_AVER_END = PARAMETERS['RDL_AVER_END']-1

    if not PARAMETERS['SHOW_PLOTS']:
        # print(PARAMETERS['SHOW_OVERLAP'])
        SHOW_PLOTS = PARAMETERS['SHOW_PLOTS']

    if PARAMETERS['CHANNELS']:
        CHANNELS = PARAMETERS['CHANNELS']

    if PARAMETERS['B_CHANNEL']:
        B_CHANNEL = PARAMETERS['B_CHANNEL']


def time_elapsed(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def file_ext_check(input_path, valid_format):
    print("checking format...")
    file_list = np.array(os.listdir(input_path))
    print(file_list)
    type_list = [[] for _ in range(len(file_list))]

    for i_file, row_file in enumerate(file_list):
        # Split the extension from the path and make it lowercase.
        name = os.path.splitext(row_file)[0]
        ext = os.path.splitext(row_file)[-1].lower()
        file_list[i_file] = name

        if ext in valid_format.keys():
            if re.search("_table", name) and ext == ".csv":
                type_list[i_file] = 'bmask_table'
                valid_format[ext] += 1
            if ext == ".tif":
                valid_format[ext] += 1
                if re.search("_Object Predictions", name):
                    type_list[i_file] = 'bmask'
                else:
                    channel_name = re.findall(" C=[a-zA-Z0-9]+", name)
                    channel = channel_name[0].replace(' C=', '')
                    if len(channel_name) == 1:
                        type_list[i_file] = channel
                        channel = int(channel)
                        if channel not in other_ch_ind_list and channel != B_CHANNEL:
                            other_ch_ind_list.append(channel)
                            # print(channel, CHANNELS)
                            try:
                                other_ch_key_list.append(CHANNELS[channel]+"_inten")
                            except Exception as err:
                                sys.exit('no channel ' + str(channel) + ' found from user setting')
                                # error_logging('high', err, 'no channel ' + str(channel) + ' found from user setting',
                                #               str(row_file))

                    if len(channel_name) != 1:
                        error_logging('high', 'not in exception', 'multiple/no channels in file name', str(row_file))
                    # if re.search(" C=0_", name):
                    #     type_list[i_file] = 'halo'
                    # if re.search(" C=1_", name):
                    #     type_list[i_file] = 'contact'
        else:
            print('wrong file format')
            error_logging('low', 'not in exception', 'wrong file format', str(row_file))

    file_table = pl.DataFrame({"file": file_list, "type": type_list})
    if len(other_ch_ind_list)+1 != len(CHANNELS.keys()):
        error_logging('high', 'not in exception', 'number of channels does not match the content in input folder', str(CHANNELS.keys()) + ' vs ' + str(other_ch_ind_list) + " with base: " + str(B_CHANNEL))
        sys.exit('number of channels does not match the content in input folder')
    print(valid_format)
    return file_table


# @time_elapsed
# def load_image(file, path):
#     #     files = file_ext_check(input_path, img_format)
#     print("loading images...")
#     print(os.path.realpath(path) + '/' + file)
#     dataset = ij.io().open(os.path.realpath(path) + '/' + file)
#     #     print(type(dataset),dataset)
#     print(type(dataset), dataset)
#     print("loading complete")
#     return dataset

# ij.ui().show(dataset)
# try:
#     for f in files:
#         dataset = ij.io().open(os.path.realpath(input_path+"/"+f))
#         ij.py.show(dataset)
# except TypeError:
#     print(TypeError)
# dataset = ij.io().open(os.path.realpath(input_path + "/" + files[0]))
# ij.ui().show(dataset)


def dump_info(image):
    """A handy function to print details of an image object."""
    name = image.name if hasattr(image, 'name') else None  # xarray
    if name is None and hasattr(image, 'getName'): name = image.getName()  # Dataset
    if name is None and hasattr(image, 'getTitle'): name = image.getTitle()  # ImagePlus
    print(f" name: {name or 'N/A'}")
    print(f" type: {type(image)}")
    print(f"dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
    print(f"shape: {image.shape}")
    print(f" dims: {image.dims if hasattr(image, 'dims') else 'N/A'}")


# def rm_xy_dup(y, x):
#     print(len(x), len(y))
#     cir = pl.DataFrame({"x": x, "y": y})
#
#     cir = cir[['x', 'y']].unique()
#
#     # for row in cir.iter_rows(named=True):
#     #     print(row['x'], row['y'])
#     print(len(cir))
#     return cir


# def reorder_xy(cir, center, r):
#     print(center)
#     cir4 = cir.filter((pl.col('y') < center[1]) & (pl.col('x') > center[0]))
#     cir4 = cir4.sort('y', 'x', descending=True)
#
#     def cir4_to_3_mirror(row, col_name):
#         return 2 * center[0] - row[col_name]
#
#     cir3_x = cir4.pipe(cir4_to_3_mirror, col_name='x')
#     cir3 = pl.DataFrame({'x': cir3_x, 'y': cir4['y']})
#     cir_down = pl.concat([cir4, pl.DataFrame({'x': [center[0]], 'y': [center[1] - r]}), cir3.reverse()])
#     print(cir3)
#     print(cir4)
#     print(cir_down)
#     cir_up_y = cir_down.select('y').pipe(
#         lambda row: 2 * center[1] - row['y'])
#     print(cir_up_y)
#     cir_up = pl.DataFrame({'x': cir_down.get_column('x'), 'y': cir_up_y})
#     cir_ord = pl.concat([pl.DataFrame({'x': [center[0] + r], 'y': [center[1]]}),
#                          cir_down,
#                          pl.DataFrame({'x': [center[0] - r], 'y': [center[1]]}),
#                          cir_up.reverse()
#                          ])
#     if len(cir) == len(cir_ord):
#         return cir_ord
#     else:
#         return cir


def erode_outline(y,x,bmap, erode):
    # print(erode)
    if bmap[y, x] == erode:
        if bmap[y+1, x] == 0 or bmap[y-1, x] == 0 or bmap[y, x+1] == 0 or bmap[y, x-1] == 0 or \
                bmap[y+1, x+1] == 0 or bmap[y-1, x-1] == 0 or bmap[y-1, x+1] == 0 or bmap[y+1, x-1] == 0:
            bmap[y, x] = ERODE_THICKNESS+2
        else:
            bmap[y, x] = erode + 1
        action_on_neighbors(y, x, True, erode_outline, bmap, erode)


def gen_outline(y, x, bmap, void_i):
    if bmap[y, x] == 1:
        bmap[y, x] = 2
        action_on_neighbors(y, x, True, gen_outline, bmap, void_i)

    if bmap[y, x] == 0:
        bmap[y, x] = 255 + void_i * 2


@time_elapsed
def gen_outline_bmask(bmask):
    bmask[bmask == 255] = 1
    arr_cen_copy = arr_cen.clone()
    void_edge = []
    for void_i, void in enumerate(arr_cen.iter_rows(named=True)):
        void_x = int(round(void['Object Center_0'], 0))
        void_y = int(round(void['Object Center_1'], 0))
        if ERODE_THICKNESS > 0:
            erode_val = int
            for e_val in range(ERODE_THICKNESS):
                # print(void_y,void_x,e_val+1)
                erode_outline(void_y, void_x, bmask, e_val+1)
                # plt.imshow(bmask)
                # plt.show()
                bmask[bmask == ERODE_THICKNESS+2] = 0
                erode_val = e_val + 1
                if bmask[void_y, void_x] == 0:
                    error_logging('low', 'not in exception', 'void erased by erosion',
                                  raw_file_name + "_id" + str(void['object_id']))
                    arr_cen_copy = arr_cen_copy.filter(pl.col('object_id') != void['object_id'])
                    break
            # print(erode_val, e_val + 2)
            bmask[bmask == (e_val + 2)] = 1

        if bmask[void_y, void_x] != 0:
            void_edge.append(255 + void_i*2)
            gen_outline(void_y, void_x, bmask, void_i)

    arr_cen_copy = arr_cen_copy.with_columns(pl.Series(name="init_edge", values=void_edge))
    # print(arr_cen_copy)
    bmask[bmask == 2] = 0
    # print(arr_cen_copy.select(pl.col('object_id')), erased, arr_cen.select(pl.col('object_id')))
    return arr_cen_copy


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If computer is running windows use cls
        command = 'cls'
    os.system(command)


def action_on_neighbors(bmask_y, bmask_x, eight, func, *arg, **kwargs):
    func(bmask_y + 1, bmask_x, *arg, **kwargs)
    func(bmask_y - 1, bmask_x, *arg, **kwargs)
    func(bmask_y, bmask_x + 1, *arg, **kwargs)
    func(bmask_y, bmask_x - 1, *arg, **kwargs)
    if eight:
        func(bmask_y + 1, bmask_x + 1, *arg, **kwargs)
        func(bmask_y - 1, bmask_x - 1, *arg, **kwargs)
        func(bmask_y + 1, bmask_x - 1, *arg, **kwargs)
        func(bmask_y - 1, bmask_x + 1, *arg, **kwargs)


def casting(bmask_y, bmask_x, bmask, cast_num):
    if bmask[bmask_y, bmask_x] == 0:
        bmask[bmask_y, bmask_x] = cast_num


# def detect_collide_ends(bmask_y, bmask_x, bmask, cast_num):


def detect_collide(bmask_y, bmask_x, bmask, edge_num, cast_num):
    if bmask[bmask_y, bmask_x] != cast_num and bmask[bmask_y, bmask_x] != edge_num and bmask[bmask_y, bmask_x] > 255:
        # print("colllide:", y_edge + 1, x_edge, b_mask[y_edge + 1, x_edge])
        overlap_xy.append([bmask_y, bmask_x])


def edge_recur(y_edge, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output, branch_data,
               count, close_flag, last_dir=None):
    # os.system('cls')
    # time.sleep(1)
    # input("press enter to continue...")
    # # clearConsole()
    # print("current y x: ", y_edge, x_edge)
    # print("starting y x: ", ring_output['ring_y'][0], ring_output['ring_x'][0])
    original_num = b_mask[y_edge, x_edge]
    # print(y_edge, x_edge, original_num)
    # casting
    if b_mask[y_edge, x_edge] == edge_num:
        action_on_neighbors(y_edge, x_edge, True, casting, b_mask, cast_num)
        # action_on_neighbors(y_edge, x_edge, True, detect_collide, b_mask, edge_num, cast_num)

    if b_mask[y_edge, x_edge] == 254:
        overlap_xy.append([y_edge, x_edge])

    record_ring(y_edge, x_edge, ring_output, branch_data, base_inten, other_inten,
                count, b_mask[y_edge, x_edge] == 254, close_flag)

    if count > 1 and ((y_edge + 1 == ring_output['ring_y'][0] and x_edge == ring_output['ring_x'][0]) or
                      (y_edge - 1 == ring_output['ring_y'][0] and x_edge == ring_output['ring_x'][0]) or
                      (y_edge == ring_output['ring_y'][0] and x_edge + 1 == ring_output['ring_x'][0]) or
                      (y_edge == ring_output['ring_y'][0] and x_edge - 1 == ring_output['ring_x'][0])):
        closed[0] = True
        # print(count, closed)

    b_mask[y_edge, x_edge] = (count % WALKER_MAX + 1)
    count += 1
    # 3rd quadrant
    if y_edge > yc and x_edge <= xc:
        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or (b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='r')

        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or (b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='d')

        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or (b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='u')

        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or (b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='l')

    # 4th quadrant
    if x_edge > xc and y_edge >= yc:
        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or (b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='u')

        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or (b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='r')

        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or (b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='l')

        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or (b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='d')

    # 1st quadrant
    if y_edge < yc and x_edge >= xc:
        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or (b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='l')

        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or (b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='u')

        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or (b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='r')

        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or (b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='d')

    # 2nd quadrant
    if x_edge < xc and y_edge <= yc:
        # go down
        if b_mask[y_edge + 1, x_edge] == edge_num or (b_mask[y_edge + 1, x_edge] == 254):
            edge_recur(y_edge + 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='d')

        # go left
        if b_mask[y_edge, x_edge - 1] == edge_num or (b_mask[y_edge, x_edge - 1] == 254):
            edge_recur(y_edge, x_edge - 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='l')

        # go up
        if b_mask[y_edge - 1, x_edge] == edge_num or (b_mask[y_edge - 1, x_edge] == 254):
            edge_recur(y_edge - 1, x_edge, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='u')

        # go right
        if b_mask[y_edge, x_edge + 1] == edge_num or (b_mask[y_edge, x_edge + 1] == 254):
            edge_recur(y_edge, x_edge + 1, b_mask, base_inten, other_inten, edge_num, cast_num, yc, xc, ring_output,
                       branch_data, count, closed, last_dir='r')

    # # collide edge
    if original_num == 254:
        if b_mask[y_edge + 1, x_edge] != cast_num and b_mask[y_edge - 1, x_edge] != cast_num and b_mask[
            y_edge, x_edge + 1] != cast_num and b_mask[y_edge, x_edge - 1] != cast_num \
                and (b_mask[y_edge + 1, x_edge] == 254 or b_mask[y_edge - 1, x_edge] == 254 or b_mask[
            y_edge, x_edge + 1] == 254 or b_mask[y_edge, x_edge - 1] == 254):
            action_on_neighbors(y_edge, x_edge, False, detect_collide, b_mask, edge_num, cast_num)
    else:
        if b_mask[y_edge + 1, x_edge] != 254 and b_mask[y_edge - 1, x_edge] != 254 and b_mask[
            y_edge, x_edge + 1] != 254 and b_mask[y_edge, x_edge - 1] != 254:
            action_on_neighbors(y_edge, x_edge, True, detect_collide, b_mask, edge_num, cast_num)
    # if b_mask[y_edge, x_edge] < 254:
    #     action_on_neighbors(y_edge, x_edge, True, detect_collide, b_mask, cast_num)


def rm_inner_casting(y, x, bmask, target):
    # print("removing...", target)
    if bmask[y, x] == target:
        bmask[y, x] = 255
        action_on_neighbors(y, x, False, rm_inner_casting, bmask, target)


def record_ring(y_edge, x_edge, ring_output, branch_output, base_inten, other_inten, count, overlap_flag, close_flag):
    # check if ring is closed
    if not closed[0]:
        if ring_output['ring_y'][count] != 0 and ring_output['ring_x'][count] != 0:
            # print("branch detected!!!!!")
            # print(ring_output['ring_y'][count], ring_output['ring_x'][count], ring_output['index'][count])
            # print(y_edge, x_edge, count)
            branch_output['index'][count] = count
            branch_output['ring_y'][count] = y_edge
            branch_output['ring_x'][count] = x_edge
            branch_output[base_inten_key][count] = base_inten[y_edge,x_edge]
            # branch_output['contact_inten'][count] = other_inten
            branch_output['overlap'][count] = overlap_flag

            for ch_ind in other_ch_ind_list:
                branch_output[CHANNELS[ch_ind] + '_inten'][count]  = other_inten[ch_ind][y_edge,x_edge]
            # if overlap_flag is True:
            #     branch_output['overlap'][count] = True
        else:
            ring_output['index'][count] = count
            ring_output['ring_y'][count] = y_edge
            ring_output['ring_x'][count] = x_edge
            ring_output[base_inten_key][count] = base_inten[y_edge,x_edge]
            # ring_output['contact_inten'][count] = other_inten
            ring_output['overlap'][count] = overlap_flag
            for ch_key in other_ch_ind_list:
                ring_output[CHANNELS[ch_key] + '_inten'][count]  = other_inten[ch_key][y_edge,x_edge]
            # if overlap_flag is True:
            #     ring_output['overlap'][count] = True
    # return close_flag


def conv(val):
    try:
        return int(val)
    except ValueError:
        return None


def plot_layer(l=0):
    fig_l, ax_l = plt.subplots(2, 2)
    if l < RING_THICKNESS:
        ax_l[0, 0].set_title(f"layer: {l + 1}")
    else:
        ax_l[0, 0].set_title("projection")

    for row_cen in arr_cen.iter_rows(named=True):
        label = f"{row_cen['object_id']}"
        ax.annotate(label,  # this is the text
                    (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                    color='red',
                    # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 0),  # distance from text to points (x,y)
                    ha='center')  # horizontal alignment can be left, right or center
        # ax.text(int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0)),
        #          f"{row_cen['object_id']}", **text_kwargs)
    ax.imshow(map_layer[l])
    plt.show()


def error_logging(err_impact, e_msg, err_type, err_source):
    print("updating error log....")
    try:
        err_log['impact'].append(err_impact)
        err_log['e_msg'].append(e_msg)
        err_log['type'].append(err_type)
        err_log['source'].append(err_source)
        pl.DataFrame(err_log).write_csv(output_path + "/err_log.csv")
    except Exception as e:
        print(e)
    print("update complete")
    return





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # change mode to headless for cluster environment
    # ij = imagej.init('sc.fiji:fiji:2.10.0', mode="interactive")
    # print(ij.getVersion())
    #
    # # HACK: # HACK: Work around ImagePlus#show() failure if no ImagePlus objects are already registered.
    # if ij.WindowManager.getIDList() is None:
    #     ij.py.run_macro('newImage("dummy", "8-bit", 1, 1, 1);')
    # plt.figure()
    f = open(output_path + '/summary.txt', 'w', newline='\n')
    f.write('version: v0.1.1\n')
    if len(sys.argv)>1:
        f.write('user settings from GUI\n')
        for item in SETTING_ITEMS.items():
            if item[0] != 'X_NORMALIZE':
                f.write(item[0] + ': ' + str(PARAMETERS[item[0]]) + '\n')
    else:
        f.write('user settings from script\n')
        for item in SETTING_ITEMS.items():
            f.write(item[0] + ': ' + str(item[1])+ '\n')

    print(INPUT_PATH, ERODE_THICKNESS, RING_THICKNESS, RDL_AVER_START, RDL_AVER_END, BK_SUB, SHOW_OVERLAP, CHANNELS)
    # check first time runner
    # bmask_path = os.path.realpath("./bmask")
    # bmask_table_path = os.path.realpath("./bmask_table")
    # droplet_path = os.path.realpath("./droplet")
    # contact_path = os.path.realpath("./contact")
    # output_path = os.path.realpath("./output_" + timestamp)
    # # print(bmask_path, bmask_table_path, droplet_path, contact_path)
    # os.makedirs(output_path)
    # check format in batch
    other_ch_ind_list = []
    other_ch_key_list = []
    file_lookup_list = file_ext_check(INPUT_PATH, INPUT_FORMAT)
    print(file_lookup_list)
    input("please confirm analysis setting and press enter to proceed...")

    ring_data_project = None
    branch_data_project = None
    normalized_data_project = None

    # load images and tables from file_lookup_list
    for item in file_lookup_list.filter(pl.col("type") == 'bmask_table').iter_rows(named=True):
        raw_file_name = item['file'].replace('_table', '')
        f.write('=====================================================================\n')
        f.write('file:' + raw_file_name + '\n')
        # lazy load centroids table
        try:
            # print(os.path.realpath(INPUT_PATH + '/' + raw_file_name + "_table.csv"))
            arr_cen = pl.scan_csv(os.path.realpath(INPUT_PATH + '/' + raw_file_name + "_table.csv")) \
                .select(ILASTIK_COLS) \
                .filter(pl.col("Predicted Class") == "LipidDroplets")
            arr_cen = arr_cen.collect()
            f.write('total voids: ' + str(arr_cen.shape[0]) + '\n')
            # print(arr_cen, arr_cen.shape)
        except Exception as e:
            print(e)
            error_logging('high', e, 'file corrupted/invalid/missing', raw_file_name + "_table.csv")
            continue

        # form outline from bmask
        try:
            map_bmask = Image.open(os.path.realpath(INPUT_PATH + '/' + raw_file_name + "_Object Predictions.tif"))
            arr_bmask = np.array(map_bmask).astype(np.uint16, casting="same_kind")
            edge_copy = arr_bmask.copy()

            arr_cen = gen_outline_bmask(edge_copy)
            f.write('voids after erosion: ' + str(arr_cen.shape[0]) + '\n')
            if len(arr_cen) == 0:
                error_logging('high', 'not in exception', 'file corrupted/invalid/missing', raw_file_name)
                continue
        except Exception as e:
            print(e)
            error_logging('high', e, 'file corrupted/invalid/missing', raw_file_name + "_Object Predictions.tif")
            continue

        # load droplet intensity
        try:
            map_droplet = Image.open(os.path.realpath(INPUT_PATH + '/' + raw_file_name + ".tif"))
            base_ch = np.array(map_droplet)
        except Exception as e:
            print(e)
            error_logging('high', e, 'file corrupted/invalid/missing', raw_file_name + ".tif")
            continue

        # # load contact intensity
        # try:
        #     # print(os.path.realpath(contact_path + '/' + raw_file_name + ".tif"))
        #     map_contact = Image.open(
        #         os.path.realpath(INPUT_PATH + '/' + raw_file_name.replace(" C=0_", " C=1_") + ".tif"))
        #     arr_contact = np.array(map_contact)
        # except Exception as e:
        #     print(e)
        #     error_logging('high', e, 'file corrupted/invalid/missing', raw_file_name.replace(" C=0_", " C=1_") + ".tif")
        #     continue
        other_chs = {}

        for ch in other_ch_ind_list:
            if int(ch) != B_CHANNEL:
                try:
                    # print(os.path.realpath(contact_path + '/' + raw_file_name + ".tif"))
                    map_inten = Image.open(
                        os.path.realpath(INPUT_PATH + '/' + raw_file_name.replace(" C=0_", " C=" + str(ch) + "_") + ".tif"))
                    other_chs[ch] = np.array(map_inten)
                    arr_contact = np.array(map_inten)
                except Exception as e:
                    print(e)
                    error_logging('high', e, 'file corrupted/invalid/missing', raw_file_name.replace(" C=0_", " C="+ str(ch) + "_") + ".tif")
                    continue

        # print(other_chs.keys())
        # view raw images and binary mask
        if SHOW_PLOTS:
            fig_input, ax_input = plt.subplots(nrows=1, ncols=1+len(other_ch_ind_list), figsize=(8*(1+len(other_ch_ind_list)), 9), layout='tight')
            fig_input.suptitle(f'input images: {raw_file_name}', fontsize=16)
            if len(other_ch_ind_list) == 0:
                ax_input.imshow(base_ch)
                ax_input.set_title(f'{CHANNELS[B_CHANNEL]} channel')
            else:
                ax_input[0].imshow(base_ch)
                ax_input[0].set_title(f'{CHANNELS[B_CHANNEL]} channel')
            # ax_input[1].imshow(arr_contact)
            for ch_i, ch in enumerate(other_ch_ind_list):
                # print(ch_i, ch,other_ch_ind_list)
                ax_input[ch_i+1].imshow(other_chs[ch])
                ax_input[ch_i+1].set_title(f'{CHANNELS[ch]} channel')
                for row_cen in arr_cen.iter_rows(named=True):
                    label = f"{row_cen['object_id']}"
                    ax_input[ch_i+1].annotate(label,  # this is the text
                                         (int(round(row_cen['Object Center_0'], 0)),
                                          int(round(row_cen['Object Center_1'], 0))),
                                         color='red',
                                         # these are the coordinates to position the label
                                         textcoords="offset points",  # how to position the text
                                         xytext=(0, 0),  # distance from text to points (x,y)
                                         ha='center')  # horizontal alignment can be left, right or center
            for row_cen in arr_cen.iter_rows(named=True):
                label = f"{row_cen['object_id']}"
                if len(other_ch_ind_list) == 0:
                    ax_input.annotate(label,  # this is the text
                                (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                                color='red',
                                # these are the coordinates to position the label
                                textcoords="offset points",  # how to position the text
                                xytext=(0, 0),  # distance from text to points (x,y)
                                ha='center')  # horizontal alignment can be left, right or center
                else:
                    ax_input[0].annotate(label,  # this is the text
                                (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                                color='red',
                                # these are the coordinates to position the label
                                textcoords="offset points",  # how to position the text
                                xytext=(0, 0),  # distance from text to points (x,y)
                                ha='center')  # horizontal alignment can be left, right or center

            plt.draw()

        t = 0

        # subtract background intensity
        if BK_SUB:
            base_1d_sort = np.sort(base_ch, axis=None)
            bk_base = np.sum(base_1d_sort[0:100]) / 100
            base_ch -= int(bk_base)
            f.write('background for ' + CHANNELS[B_CHANNEL] + ': ' + str(bk_base) + '\n')

            # contact_1d_sort = np.sort(arr_contact, axis=None)
            # background_contact = np.sum(contact_1d_sort[0:100]) / 100
            # arr_contact -= int(background_contact)

            for ch in other_ch_ind_list:
                other_ch_1d_sort = np.sort(other_chs[ch], axis=None)
                bk_other_ch = np.sum(other_ch_1d_sort[0:100]) / 100
                other_chs[ch] -= int(bk_other_ch)
                f.write('background for ' + CHANNELS[ch] + ': ' + str(bk_other_ch) + '\n')

            # bk_base = np.median(base_ch)
            # base_ch -= int(bk_base)
            # print(bk_base)
            # f.write('background for ' + CHANNELS[B_CHANNEL] + ': ' + str(bk_base) + '\n')

            # for ch in other_ch_ind_list:
            #     bk_other_ch = np.median(other_chs[ch])
            #     other_chs[ch] -= int(bk_other_ch)
            #     print(bk_other_ch)
            #     f.write('background for ' + CHANNELS[ch] + ': ' + str(bk_other_ch) + '\n')

        plot_y_max_droplet = 0
        plot_y_max_other_chs = 0
        # print(background_contact,background_droplet)
        map_layer = np.full((RING_THICKNESS + 1, np.shape(edge_copy)[0], np.shape(edge_copy)[1]), 0)
        ring_data_image = None
        branch_data_image = None

        base_inten_key = CHANNELS[B_CHANNEL] + '_inten'
        # other_ch_names = [[CHANNELS[key]] for key in CHANNELS.keys()]

        # print(arr_cen)
        while t < RING_THICKNESS:
            for i, row in enumerate(arr_cen.iter_rows(named=True)):
                cen_x = int(round(row['Object Center_0'], 0))
                cen_y = int(round(row['Object Center_1'], 0))

                # print("i:", i)
                # if edge_copy[cen_y,cen_x] == 0:
                #     print('void erased by erosion')
                #     continue
                # allocate table for ring data
                ring_len = int(row['Size in pixels']) * 10  # could be too small
                # print(row['object_id'], ring_len)
                overlap_xy = []
                closed = [False]
                ring_data = {'image': np.full((ring_len,), raw_file_name),
                             'object_id': np.full((ring_len,), row['object_id'], dtype=np.uint16),
                             'layer': np.full((ring_len,), t + 1, dtype=np.uint16),
                             'index': np.zeros((ring_len,), dtype=np.uint16),
                             'ring_y': np.zeros((ring_len,), dtype=np.uint16),
                             'ring_x': np.zeros((ring_len,), dtype=np.uint16),
                             'overlap': np.full((ring_len,), False, dtype=bool),
                             'closed': np.full((ring_len,), False, dtype=bool),
                             base_inten_key: np.zeros((ring_len,), dtype=np.uint16)}

                for key in other_ch_ind_list:
                    ring_data[CHANNELS[key]+'_inten'] = np.zeros((ring_len,), dtype=np.uint16)

                branch_data = copy.deepcopy(ring_data)

                # print(ring_data.keys(), branch_data.keys())
                # input('....')
                # branch_data = {
                #     'image': np.full((ring_len,), raw_file_name),
                #     'object_id': np.full((ring_len,), row['object_id'], dtype=np.uint16),
                #     'layer': np.full((ring_len,), t + 1, dtype=np.uint16),
                #     'index': np.zeros((ring_len,), dtype=np.uint16),
                #     'ring_y': np.zeros((ring_len,), dtype=np.uint16),
                #     'ring_x': np.zeros((ring_len,), dtype=np.uint16),
                #     # 'ring_inten': np.zeros((ring_len,), dtype=np.uint16),
                #     # 'contact_inten': np.zeros((ring_len,), dtype=np.uint16),
                #     'overlap': np.full((ring_len,), False, dtype=bool),
                #     'closed': np.full((ring_len,), False, dtype=bool)
                # }

                # track centroid for each void


                # edge = 255 + t + 2 * i
                # print(row)
                edge = row['init_edge'] + t
                cast = edge + 1

                if t == 0:
                    y_Start = cen_y
                else:
                    start_last_layer = ring_data_image.filter(
                        (pl.col("object_id") == row['object_id']) & (pl.col("layer") == t) & (pl.col("index") == 0))
                    # print(start_last_layer)
                    y_Start = start_last_layer['ring_y'][0]

                # removing inner casting
                if t == 1:
                    # print('removing inner casting...')
                    rm_inner_casting(y_Start - 1, cen_x, edge_copy, edge)

                # print("starting(y, x):", y_Start, cen_x)

                # locate starting pixel
                try:
                    while not (edge_copy[y_Start, cen_x] == edge or edge_copy[y_Start, cen_x] == 254):
                        y_Start += 1
                        # print(edge_copy[y_Start, cen_x], edge, y_Start, cen_x)
                except Exception as e:
                    print(e)
                    error_logging('high', e, 'fail to locate edge', raw_file_name + "_layer" + str(t) + "_id" + str(row['object_id']))
                    continue

                # print(type(ring_data['layer']))
                # print("void info:", y_Start, cen_x, edge, cast, edge_copy[y_Start, cen_x], row['object_id'])

                try:
                    edge_recur(y_Start, cen_x, edge_copy, base_ch, other_chs, edge, cast, cen_y, cen_x, ring_data,
                               branch_data, 0,
                               closed)  # y_edge, x_edge, b_mask, ring_map, contact_map, edge_num, cast_num, yc, xc, ring_output, count, last_dir= None
                except Exception as e:
                    error_logging('medium', e, 'pre-allocated array size too small',
                                  raw_file_name + "_layer" + str(t) + "_id" + str(row['object_id']))
                    continue

                # patch overlapped spots
                b4_patch = edge_copy.copy()
                for pix in overlap_xy:
                    edge_copy[pix[0], pix[1]] = 254

                # cook droplet data per layer and swap branched pixel
                for i_branch in reversed(range(len(branch_data['index']))):
                    dup_i = branch_data['index'][i_branch]
                    # overflow
                    if (ring_data['ring_y'][dup_i + 1] == branch_data['ring_y'][i_branch] and (ring_data['ring_x'][dup_i + 1] + 1 == branch_data['ring_x'][i_branch] or ring_data['ring_x'][dup_i + 1] - 1 == branch_data['ring_x'][i_branch])) or\
                        (ring_data['ring_x'][dup_i + 1] == branch_data['ring_x'][i_branch] and (
                                    ring_data['ring_y'][dup_i + 1] + 1 == branch_data['ring_y'][i_branch] or
                                    ring_data['ring_y'][dup_i + 1] - 1 == branch_data['ring_y'][i_branch])):
                    # if (-1 <= int(ring_data['ring_y'][dup_i + 1]) - int(branch_data['ring_y'][i_branch]) <= 1) and (
                    #         -1 <= int(ring_data['ring_x'][dup_i + 1]) - int(branch_data['ring_x'][i_branch]) <= 1):
                        print("swap branch data.....")
                        # print(ring_data['ring_y'][dup_i + 1], ring_data['ring_x'][dup_i + 1],
                        #       branch_data['ring_y'][i_branch], branch_data['ring_x'][i_branch])
                        swap_list = ['ring_y','ring_x', base_inten_key]
                        for ch_key in other_ch_ind_list:
                            swap_list.append(CHANNELS[ch_key] + '_inten')

                        temp = {}
                        # print(swap_list)
                        for k in swap_list:
                            # print(k, ring_data[k][dup_i])
                            temp[k] = ring_data[k][dup_i]
                            ring_data[k][dup_i] = branch_data[k][i_branch]
                            branch_data[k][i_branch] = temp[k]

                # check ring if closed and decide the flag
                if closed[0]:
                    ring_data['closed'] = np.full((ring_len,), True, dtype=bool)
                else:
                    error_logging('low', 'not in exception', 'open ring',
                                  raw_file_name + "_layer" + str(t) + "_id" + str(row['object_id']))

                ring_data = pl.DataFrame(ring_data).filter((pl.col('ring_y') != 0) & (pl.col('ring_y') != 0))
                branch_data = pl.DataFrame(branch_data).filter((pl.col('ring_y') != 0) & (pl.col('ring_y') != 0))
                # print(ring_data, branch_data)
                if ring_data_image is None:
                    ring_data_image = ring_data.clone()
                else:
                    ring_data_image = pl.concat([ring_data_image, ring_data])

                if branch_data_image is None:
                    branch_data_image = branch_data.clone()
                else:
                    branch_data_image = pl.concat([branch_data_image, branch_data])

                if ring_data_project is None:
                    ring_data_project = ring_data.clone()
                else:
                    ring_data_project = pl.concat([ring_data_project, ring_data])

                if branch_data_project is None:
                    branch_data_project = branch_data.clone()
                else:
                    branch_data_project = pl.concat([branch_data_project, branch_data])

                # # save and export as csv
                # ring_data_layer.write_csv(os.path.realpath(output_path + '/ring.csv'))
                # branch_data_layer.write_csv(os.path.realpath(output_path + '/branch.csv'))

                # plot bmask
                if t >= RING_THICKNESS:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle(f"droplet id: {ring_data['object_id'][0]} , layer: {ring_data['layer'][0]}")

                    ax1.imshow(edge_copy)
                    text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1')
                    for row_cen in arr_cen.iter_rows(named=True):
                        # label = f"{row_cen['object_id']}"
                        ax1.text(int(round(row_cen['Object Center_0'], 0)),
                                 int(round(row_cen['Object Center_1'], 0)),
                                 f"{row_cen['object_id']}", **text_kwargs)
                        # ax1.annotate(label,  # this is the text
                        #              (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),  # these are the coordinates to position the label
                        #              textcoords="offset points",  # how to position the text
                        #              xytext=(0, 0),  # distance from text to points (x,y)
                        #              ha='center')  # horizontal alignment can be left, right or center
                    ax2.imshow(b4_patch)

                    # label = f"{row['object_id']}"
                    ax2.text(cen_x, cen_y, f"{row['object_id']}", **text_kwargs)
                    plt.show()

            # print(ring_data_image, branch_data_image)


            # save and export as csv
            # print(ring_data_project)
            try:
                ring_data_project.write_csv(os.path.realpath(output_path + '/ring_' + timestamp + '.csv'))
                branch_data_project.write_csv(os.path.realpath(output_path + '/branch_' + timestamp + '.csv'))
            except Exception as e:
                print(e)
                error_logging('high', e, 'empty datatable due to excessive erosion', raw_file_name + ".tif")

            if SHOW_PLOTS:
                map_layer[t] = edge_copy
                map_filtered = edge_copy.copy()
                map_filtered[map_filtered < 254] = 0
                map_filtered[map_filtered >= 254] = (t + 1) * 255
                map_layer[-1] += map_filtered

            # increase thickness
            t += 1

        plot_y_max_droplet = np.sort(ring_data_image[base_inten_key].to_numpy(), axis=None)[-1]
        plot_y_max_droplet = plot_y_max_droplet//10 + plot_y_max_droplet
        plot_y_max_other_chs = 0
        for k in other_ch_key_list:
            ch_max = np.sort(ring_data_image[k].to_numpy(), axis=None)[-1]
            if ch_max > plot_y_max_other_chs:
                plot_y_max_other_chs = ch_max
        plot_y_max_other_chs = plot_y_max_other_chs//10 + plot_y_max_other_chs
        # input("press enter")

        plot_data_void = {
            'overlap': list,
            base_inten_key: list
        }

        for ch_key in other_ch_ind_list:
            plot_data_void[CHANNELS[ch_key] + '_inten'] = []
        plot_data = [[copy.deepcopy(plot_data_void) for _ in range(RING_THICKNESS + 1)] for _ in range(arr_cen.shape[0])]

        # swap overlapped pixels with NaN
        ring_data_image_clone = ring_data_image.clone()

        for key in plot_data_void.keys():
            if key != "overlap":
                ring_data_image_clone = ring_data_image_clone.with_columns([
                    pl.when(pl.col("overlap") == 1).then(pl.lit(float("NaN"))).otherwise(
                        pl.col(key)).keep_name(),
                ])


        # # print(ring_data_image_clone)
        # ring_data_image_clone.write_csv(os.path.realpath(output_path + '/test.csv'))
        # input("press enter")

        for i, row in enumerate(arr_cen.iter_rows(named=True)):
            if SHOW_OVERLAP:
                layer_data = ring_data_image_clone \
                    .filter((pl.col("object_id") == row["object_id"])) \
                    .groupby("layer", maintain_order=True).all()
            else:
                layer_data = ring_data_image_clone \
                    .filter((pl.col("overlap") == 0) & (pl.col("object_id") == row["object_id"])) \
                    .groupby("layer", maintain_order=True).all()

            ring_data = layer_data[base_inten_key].to_list()
            other_chs_plot_data = {}
            for k in other_ch_key_list:
                other_chs_plot_data[k] = layer_data[k].to_list()
            index_data = layer_data["index"].to_list()
            overlap_data = layer_data["overlap"].to_list()

            other_chs_inter = {}
            normalized_data_img = {}
            normalized_data_img[base_inten_key + '_sum'] = np.full((X_NORMALIZE,), np.nan)
            normalized_data_img[base_inten_key + '_rdd_aver'] = np.full((X_NORMALIZE,), np.nan)
            for k in other_ch_key_list:
                normalized_data_img[k + '_sum'] = np.full((X_NORMALIZE,), np.nan)
                normalized_data_img[k + '_rdl_aver'] = np.full((X_NORMALIZE,), np.nan)

            for x in range(RING_THICKNESS):
                normalized_data_img['img'] = np.full((X_NORMALIZE,), item['file'])
                normalized_data_img['object_id'] = np.full((X_NORMALIZE,), row['object_id'])
                normalized_data_img['layer'] = np.full((X_NORMALIZE,), x)
                ring_inter = interpolate.interp1d(index_data[x], ring_data[x])

                for k in other_ch_key_list:
                    other_chs_inter[k] = interpolate.interp1d(index_data[x], other_chs_plot_data[k][x])
                new_x = np.linspace(index_data[x][0], index_data[x][-1], X_NORMALIZE)
                plot_data[i][x][base_inten_key] = ring_inter(new_x)
                normalized_data_img[base_inten_key] = ring_inter(new_x)
                # plot_data[i][x]['contact_inten'] = contact_inter(new_x)
                for k in other_ch_key_list:
                    plot_data[i][x][k] = other_chs_inter[k](new_x)
                    normalized_data_img[k] = other_chs_inter[k](new_x)

                # set up compression intensity
                if RDL_AVER_END > RDL_AVER_START and RDL_AVER_START <= x <= RDL_AVER_END:
                    # initialize first compression layer
                    if x == RDL_AVER_START:
                        plot_data[i][-1][base_inten_key] = np.array(plot_data[i][x][base_inten_key])
                        # plot_data[i][-1]['contact_inten'] = np.array(plot_data[i][x]['contact_inten'])
                        for k in other_ch_key_list:
                            plot_data[i][-1][k] = np.array(plot_data[i][x][k])
                    else:
                        plot_data[i][-1][base_inten_key] = np.add(plot_data[i][-1][base_inten_key],
                                                                np.array(plot_data[i][x][base_inten_key]))
                    # plot_data[i][-1]['contact_inten'] = np.add(plot_data[i][-1]['contact_inten'],
                    #                                            np.array(plot_data[i][x]['contact_inten']))
                        for k in other_ch_key_list:
                            plot_data[i][-1][k] = np.add(plot_data[i][-1][k],
                                                                      np.array(plot_data[i][x][k]))
                    # print(np.nansum(plot_data[i][-1]['ring_inten']), np.nansum(plot_data[i][-1]['contact_inten']))
                    if x == RDL_AVER_END:
                        normalized_data_img[base_inten_key + '_sum'] = copy.deepcopy(plot_data[i][-1][base_inten_key])
                        plot_data[i][-1][base_inten_key] /= (RDL_AVER_END - RDL_AVER_START + 1)
                        normalized_data_img[base_inten_key+'_rdd_aver'] = plot_data[i][-1][base_inten_key]

                        # plot_data[i][-1]['contact_inten'] /= (RDL_AVER_END - RDL_AVER_START + 1)
                        for k in other_ch_key_list:
                            normalized_data_img[k + '_sum'] = copy.deepcopy(plot_data[i][-1][k])
                            plot_data[i][-1][k] /= (RDL_AVER_END - RDL_AVER_START + 1)
                            normalized_data_img[k + '_rdl_aver'] = plot_data[i][-1][k]

                # if no compression, copy the value of last layer for compression layer
                if RDL_AVER_END <= RDL_AVER_START:
                    # print('no compression')
                    plot_data[i][-1][base_inten_key] = np.full((X_NORMALIZE,), np.nan)
                    # plot_data[i][-1]['contact_inten'] = np.full((X_NORMALIZE,), np.nan)
                    for k in other_ch_key_list:
                        plot_data[i][-1][k] = np.full((X_NORMALIZE,), np.nan)

                if normalized_data_project is None:
                    normalized_data_project = pl.DataFrame(normalized_data_img)

                else:
                    # print(normalized_data_img.keys())
                    normalized_data_project = pl.concat([normalized_data_project, pl.DataFrame(normalized_data_img)])
        # normalized_data_project = pl.concat([normalized_data_project, normalized_data_img])
        if RDL_AVER_END > RDL_AVER_START:
            normalized_data_project.write_csv(os.path.realpath(output_path + '/normalized_ring_' + str(RDL_AVER_START) + "TO" + str(RDL_AVER_END) + "_" + timestamp + '.csv'))

        if SHOW_PLOTS:
            def show_layer(layer):
                return map_layer[layer - 1]

            def show_ring_plots(layer, droplet_index):
                # print(droplet_index, layer - 1, plot_data[droplet_index][layer - 1]['ring_inten'][0])
                return plot_data[droplet_index][layer - 1][base_inten_key]

            # def show_other_chs_plots(layer, droplet_index, channel):
            #     print(channel)
            #     # print(droplet_index, layer - 1, plot_data[droplet_index][layer - 1]['ring_inten'][0])
            #     # ch_plots = []
            #     # for ch_k in other_ch_key_list:
            #     #     ch_plots.append(plot_data[droplet_index][layer - 1][ch_k])
            #     # print(len(ch_plots))
            #     # return ch_plots
            #     return plot_data[droplet_index][layer - 1][channel]
            #     # return plot_data[droplet_index][layer - 1]['contact_inten']

            def show_other_chs_plots(layer, droplet_index,channel):
                # print(droplet_index, layer - 1, plot_data[droplet_index][layer - 1]['ring_inten'][0])
                # ch_plots = []
                # for ch_k in other_ch_key_list:
                #     ch_plots.append(plot_data[droplet_index][layer - 1][ch_k])
                # print(len(ch_plots))
                # return ch_plots
                return plot_data[droplet_index][layer - 1][other_ch_key_list[channel]]
                # return plot_data[droplet_index][layer - 1]['contact_inten']


            # plotting map
            fig, ax = plt.subplots()
            plt.subplots_adjust(top=0.9)

            for row_cen in arr_cen.iter_rows(named=True):
                label = f"{row_cen['object_id']}"
                ax.annotate(label,  # this is the text
                            (int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0))),
                            color='red',
                            # these are the coordinates to position the label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 0),  # distance from text to points (x,y)
                            ha='center')  # horizontal alignment can be left, right or center
                # ax.text(int(round(row_cen['Object Center_0'], 0)), int(round(row_cen['Object Center_1'], 0)),
                #          f"{row_
            axfreq = plt.axes([0.1, 0.95, 0.1, 0.01])  # right, top, length, width
            slider_layer = Slider(axfreq, label="layer  ", valmin=1, valmax=RING_THICKNESS + 1, valstep=1)

            controls = iplt.imshow(show_layer, layer=slider_layer, ax=ax)
            fig.suptitle(f"traverse map")
            plt.get_current_fig_manager().window.setGeometry(0, 0, 640, 480)

            # plotting intensity for the whole layer
            controls_plot = [[None for _ in range(len(CHANNELS.keys()))] for _ in range(arr_cen.shape[0])]
            # controls_plot = [[] for _ in range(len(CHANNELS.keys())) for _ in range(arr_cen.shape[0])]
            c = int(math.ceil(np.sqrt(arr_cen.shape[0])))
            r = int(arr_cen.shape[0] / c + 1)
            # print(c, r, controls_plot)
            fig, ax2 = plt.subplots(r, c, sharex=True, sharey=True, figsize=(4*c, 3*r))
            fig.suptitle(f"intensity plot for image: {raw_file_name}")
            fig.subplots_adjust(hspace=0, wspace=0, top=0.95, bottom=0.05, right=0.95, left=0.05)
            lines_ax_twin = []
            labels_ax_twin = []

            for i, row in enumerate(arr_cen.iter_rows(named=True)):
                table_row = i // c
                table_col = i % c
                axfreq_droplet = plt.axes([1, 0.95, 0.1, 0.01])  # right, top, length, width
                slider_droplet = Slider(axfreq_droplet, label="", valmin=i, valmax=i, valstep=1)
                ax_twin = ax2[table_row, table_col].twinx()
                controls_plot[i][0] = iplt.plot(show_ring_plots, label=base_inten_key.replace('_inten',''), layer=slider_layer, droplet_index=slider_droplet,
                                                ax=ax2[table_row, table_col])
                for k_i, k in enumerate(other_ch_key_list):
                    axfreq_ch = plt.axes([1, 0.95, 0.1, 0.01])  # right, top, length, width
                    slider_ch = Slider(axfreq_ch, label="", valmin=k_i, valmax=k_i, valstep=1)
                    controls_plot[i][1] = iplt.plot(show_other_chs_plots, label=k.replace('_inten',''), color=color_list[k_i%9], layer=slider_layer,
                                                    droplet_index=slider_droplet, channel=slider_ch, ax=ax_twin)

                ax2[table_row, table_col].set_ylim([0, plot_y_max_droplet])
                ax2[table_row, table_col].set_title(f"object_id: {row['object_id']}", y=1.0, pad=-14)
                ax2[table_row, table_col].set_xlabel('distance(normalized)')
                if i == c - 1:
                    ax2[table_row, table_col].legend(loc="upper left")
                    ax_twin.legend(loc='upper right')
                    # plt.legend(handles=lines_ax_twin, loc='upper right')
                    # fig.legend()
                if table_col == 0:
                    ax2[table_row, table_col].set_ylabel('raw intensity')
                ax_twin.set_ylim([0, plot_y_max_other_chs])
                ax2[table_row, table_col].tick_params(axis="y", labelcolor="#1f77b4")
                if table_col != c - 1:
                    ax_twin.yaxis.set_visible(False)
                else:
                    ax_twin.tick_params(axis="y", labelcolor="red")

            plt.show()

        c = int(math.ceil(np.sqrt(arr_cen.shape[0])))
        r = int(arr_cen.shape[0] / c + 1)
        for x in range(RING_THICKNESS+1):
            if (RDL_AVER == 0 and x == RING_THICKNESS) or (RING_THICKNESS == 1 and x == RING_THICKNESS):
                break
            fig_layer, ax_layer = plt.subplots(nrows=r, ncols=c, sharex=True, sharey=True, figsize=(c*4, r*3))
            if x == RING_THICKNESS:
                fig_layer.suptitle(f"intensity plot for layer: Radial Average")
            else:
                fig_layer.suptitle(f"intensity plot for layer: {x+1}")
            fig_layer.subplots_adjust(hspace=0, wspace=0, top=0.95, bottom=0.05, right=0.95, left=0.05)
            for i, row in enumerate(arr_cen.iter_rows(named=True)):

                table_row = i // c
                table_col = i % c
                ax_layer_twin = ax_layer[table_row, table_col].twinx()
                ax_layer[table_row, table_col].plot(plot_data[i][x][base_inten_key], label=base_inten_key.replace('_inten',''))
                # ax_layer_twin.plot(plot_data[i][x]['contact_inten'], label="FABCCON", color="red")
                for k_i, k in enumerate(other_ch_key_list):
                    ax_layer_twin.plot(plot_data[i][x][k], label=k.replace('_inten',''), color=color_list[k_i%9])

                ax_layer[table_row, table_col].set_ylim([0, plot_y_max_droplet])
                ax_layer[table_row, table_col].set_title(f"object_id: {row['object_id']}", y=1.0, pad=-14)
                ax_layer[table_row, table_col].set_xlabel('distance (px)')
                if i == c - 1:
                    ax_layer[table_row, table_col].legend(loc="upper left")
                    ax_layer_twin.legend(loc='upper right')
                    # fig.legend()
                if table_col == 0:
                    ax_layer[table_row, table_col].set_ylabel('raw intensity')
                ax_layer_twin.set_ylim([0, plot_y_max_other_chs])
                ax_layer[table_row, table_col].tick_params(axis="y", labelcolor="#1f77b4")
                if table_col != c - 1:
                    ax_layer_twin.yaxis.set_visible(False)
                else:
                    ax_layer_twin.tick_params(axis="y", labelcolor="red")
            plt.savefig(os.path.realpath(output_path + '/' + raw_file_name + '_layer' + str(x+1) + '_' + timestamp + '.png'))
            plt.close(fig_layer)

        print("==============================================NEW IMAGE=================================================")
        try:
            ring_data_image.write_csv(os.path.realpath(output_path+'/'+raw_file_name+'_'+ timestamp + '.csv'))
        except Exception as e:
            print(e)
            error_logging('high', e, 'empty table', raw_file_name + ".tif")
    # print(ring_data_project.shape)
    f.close()
