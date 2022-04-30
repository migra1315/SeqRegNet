import os
import os.path
import random
import time

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data as data
import yaml
from scipy import interpolate

import model


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def init_model(model_name):
    if model_name == 'ucr':
        Model = model.RegNet
        print('init model RegNet Ucr ...')
    elif model_name == 'fucr':
        Model = model.RegNet_full
        print('init model RegNet Fucr ...')

    elif model_name == 'cr':
        Model = model.RegNet_CR
        print('init model RegNet Cr ...')

    else:
        raise ValueError("model name should be cat or cr")
    return Model


def make_ready(args, config):
    case = args.case_num
    if args.remote:
        data_folder = f'/data/JY/Dirlab/case{case}/'
        landmark_file = f'/data/JY/Dirlab/case{case}/Case{case}_300_00_50.pt'
    else:
        data_folder = f'../DataBase/Dirlab/case{case}/'
        landmark_file = f'../DataBase/Dirlab/case{case}/Case{case}_300_00_50.pt'
    crop_range, pixel_spacing = get_case(case)
    input_image, image_shape, num_image = load_data(data_folder, crop_range)
    if config['group_index_list'] is not None:
        input_image = input_image[config['group_index_list']]

    # 导入标记点，后续计算TRE
    landmark_info = torch.load(landmark_file)
    landmark_00 = landmark_info['landmark_00']
    landmark_50 = landmark_info['landmark_50']

    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    landmark_00_converted = np.flip(landmark_00, axis=1) - np.array(
        [crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype=np.float32)
    landmark_50_converted = np.flip(landmark_50, axis=1) - np.array(
        [crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype=np.float32)

    return case, input_image, pixel_spacing, grid_tuple, landmark_00_converted, landmark_50_converted


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


class StopCriterion(object):
    def __init__(self, stop_std=0.001, query_len=100, num_min_iter=200):
        self.query_len = query_len
        self.stop_std = stop_std
        self.loss_list = []
        self.loss_min = 1.
        self.num_min_iter = num_min_iter

    def add(self, loss):
        self.loss_list.append(loss)
        if loss < self.loss_min:
            self.loss_min = loss
            self.loss_min_i = len(self.loss_list)

    def stop(self):
        # return True if the stop creteria are met
        query_list = self.loss_list[-self.query_len:]
        query_std = np.std(query_list)
        if query_std < self.stop_std and self.loss_list[-1] - self.loss_min < self.stop_std / 3. and len(
                self.loss_list) > self.loss_min_i and len(self.loss_list) > self.num_min_iter:
            return True
        else:
            return False


class CalTRE:
    '''
    pairwise方法
    根据形变场计算配准后的TRE
    构建一个规范网格，及网格上各点的位移场，根据此采样LandMark的形变场
    '''

    def __init__(self, grid_tuple, disp_f2m):
        self.dim = 3
        self.spatial_transformer = model.SpatialTransformer(dim=self.dim)
        '''
        STN网络中用到的都是反向映射，即warpped中(x,y,z)处的点来自moving的哪一处
        WARPPED_INT(x,y,x) = MVOING_INT(x-disp_x, y-disp_y, z-disp_z)
        正向映射才是我们计算TRE需要的形变场，反映了mving中的点经过形变场后到了哪一个位置
        '''
        disp_m2f = self.inverse_disp(disp_f2m)
        print(disp_m2f.size())
        '''
        一个采样器，给出一个3维网格，和网格上的数据点 -> 也就是各处的形变场
        '''
        self.inter = interpolate.RegularGridInterpolator(grid_tuple,
                                                         np.moveaxis(disp_m2f.detach().cpu().numpy(), 0, -1))

    def inverse_disp(self, disp, threshold=0.01, max_iteration=20):
        '''
        compute the inverse field. implementation of "A simple fixed‐point approach to invert a deformation field"

        disp : (2, h, w) or (3, d, h, w)
            displacement field
        '''
        forward_disp = disp.detach().to(device='cuda')
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)
        backward_disp = torch.zeros_like(forward_disp)
        backward_disp_old = backward_disp.clone()
        for i in range(max_iteration):
            backward_disp = -self.spatial_transformer(forward_disp, backward_disp)
            diff = torch.max(torch.abs(backward_disp - backward_disp_old)).item()
            if diff < threshold:
                break
            backward_disp_old = backward_disp.clone()
        if disp.ndim < self.dim + 2:
            backward_disp = torch.squeeze(backward_disp, 0)

        return backward_disp

    def cal_disp(self, landmark_moving, landmark_fixed, spacing):
        diff_list = []
        # gt = np.flip((landmark_fixed[1] - landmark_moving[1]), 0)  # 对应的方向分别为[240,157,83]
        # pred = self.inter(landmark_moving[1])

        for i in range(300):
            # landmark_moving[i]处的推理形变场pred
            # landmark_moving[i]处的真实形变场gt
            pred = self.inter(landmark_moving[i])
            gt = np.flip((landmark_fixed[i] - landmark_moving[i]), 0)  # 对应的方向分别为[240,157,83]
            diff_list.append(pred - gt)
        diff_voxel = np.array(diff_list).squeeze(1)
        # 计算300个点对的欧氏距离
        diff = (np.sum((diff_voxel * spacing) ** 2, 1)) ** 0.5
        return np.mean(diff), np.std(diff), diff


class CalTRE_2:
    """
    :param
        grid_tuple 与形变场大小一致的规则网格，用于
        disp_f2m
    :return
    """

    def __init__(self, grid_tuple, disp_f2m):
        self.dim = 3
        self.spatial_transformer = model.SpatialTransformer(dim=self.dim)

        disp05 = self.spatial_transformer(disp_f2m[1:2], disp_f2m[2:3]) + disp_f2m[2:3]  # (n, 3, d, h, w)
        disp05 = self.spatial_transformer(disp05, disp_f2m[3:4]) + disp_f2m[3:4]  # (n, 3, d, h, w)
        disp05 = self.spatial_transformer(disp05, disp_f2m[4:5]) + disp_f2m[4:5]  # (n, 3, d, h, w)
        disp05 = self.spatial_transformer(disp05, disp_f2m[5:6]) + disp_f2m[5:6]  # (n, 3, d, h, w)
        '''
        STN网络中用到的都是反向映射，即warpped中(x,y,z)处的点来自moving的哪一处
        WARPPED_INT(x,y,x) = MVOING_INT(x-disp_x, y-disp_y, z-disp_z)
        正向映射才是我们计算TRE需要的形变场，反映了mving中的点经过形变场后到了哪一个位置
        '''
        disp_m2f = self.inverse_disp(disp05)
        '''
        一个采样器，给出一个3维网格，和网格上的数据点 -> 也就是各处的形变场
        '''
        self.inter = interpolate.RegularGridInterpolator(grid_tuple,
                                                         np.moveaxis(disp_m2f.squeeze(0).detach().cpu().numpy(), 0, -1))

    def inverse_disp(self, disp, threshold=0.01, max_iteration=20):
        '''
        compute the inverse field. implementation of "A simple fixed‐point approach to invert a deformation field"

        disp : (2, h, w) or (3, d, h, w)
            displacement field
        '''
        forward_disp = disp.detach().to(device='cuda')
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)
        backward_disp = torch.zeros_like(forward_disp)
        backward_disp_old = backward_disp.clone()
        for i in range(max_iteration):
            backward_disp = -self.spatial_transformer(forward_disp, backward_disp)
            diff = torch.max(torch.abs(backward_disp - backward_disp_old)).item()
            if diff < threshold:
                break
            backward_disp_old = backward_disp.clone()
        if disp.ndim < self.dim + 2:
            backward_disp = torch.squeeze(backward_disp, 0)

        return backward_disp

    def cal_disp(self, landmark_moving, landmark_fixed, spacing):
        diff_list = []
        # gt = np.flip((landmark_fixed[1] - landmark_moving[1]), 0)  # 对应的方向分别为[240,157,83]
        # pred = self.inter(landmark_moving[1])

        for i in range(300):
            # landmark_moving[i]处的推理形变场pred
            # landmark_moving[i]处的真实形变场gt
            pred = self.inter(landmark_moving[i])
            gt = np.flip((landmark_fixed[i] - landmark_moving[i]), 0)  # 对应的方向分别为[240,157,83]
            diff_list.append(pred - gt)
        diff_voxel = np.array(diff_list).squeeze(1)
        # 计算300个点对的欧氏距离
        diff = (np.sum((diff_voxel * spacing) ** 2, 1)) ** 0.5
        return np.mean(diff), np.std(diff), diff


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_case(case):
    if case == 1:
        crop_range = [slice(0, 83), slice(43, 200), slice(10, 250)]
        pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
    elif case == 2:
        crop_range = [slice(5, 98), slice(30, 195), slice(8, 243)]
        pixel_spacing = np.array([1.16, 1.16, 2.5], dtype=np.float32)
    elif case == 3:
        crop_range = [slice(0, 95), slice(42, 209), slice(10, 248)]
        pixel_spacing = np.array([1.15, 1.15, 2.5], dtype=np.float32)
    elif case == 4:
        crop_range = [slice(0, 90), slice(45, 209), slice(11, 242)]
        pixel_spacing = np.array([1.13, 1.13, 2.5], dtype=np.float32)
    elif case == 5:
        crop_range = [slice(0, 90), slice(60, 222), slice(16, 237)]
        pixel_spacing = np.array([1.10, 1.10, 2.5], dtype=np.float32)
    elif case == 6:
        crop_range = [slice(10, 107), slice(144, 328), slice(132, 426)]
        pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
    elif case == 7:
        crop_range = [slice(13, 108), slice(141, 331), slice(114, 423)]
        pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
    elif case == 8:
        # crop_range = [slice(17, 113), slice(91, 299), slice(123, 380)]
        # pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
        crop_range = [slice(18, 118), slice(84, 299), slice(113, 390)]
        pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
    elif case == 9:
        crop_range = [slice(0, 70), slice(126, 334), slice(128, 390)]
        pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
    elif case == 10:
        crop_range = [slice(0, 90), slice(119, 333), slice(140, 382)]
        pixel_spacing = np.array([0.97, 0.97, 2.5], dtype=np.float32)
    else:
        raise ValueError('case num should be 0 to 10')
    return crop_range, pixel_spacing


def load_data(data_folder, crop_range):
    # 导入数据集，尺寸: [10, 1, 94, 256, 256]/归一化/裁切
    image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if
                              (file_name.lower().endswith('mhd') or file_name.lower().endswith('nii'))])
    image_list = []
    image_list = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, file_name))) for file_name in
                  image_file_list]
    input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
    input_image = input_image[:, :, crop_range[0], crop_range[1], crop_range[2]]
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    num_image = input_image.shape[0]  # number of image in the group
    image_shape = input_image.size()[2:]
    return input_image, image_shape, num_image


def load_data_test(data_folder, crop_range):
    # 导入数据集，尺寸: [10, 1, 94, 256, 256]/归一化/裁切
    image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('mhd')])
    image_list = []
    image_list = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, file_name))) for file_name in
                  image_file_list]
    input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
    # input_image = input_image[:, :, crop_range[0], crop_range[1], crop_range[2]]
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    image_shape = input_image.size()[2:]  # (d, h, w)
    num_image = input_image.shape[0]  # number of image in the group
    return input_image, image_shape, num_image


def write_update_loss(writer, simi_loss, smooth_loss, jdet_loss, cyclic_loss, total_loss, i):
    writer.add_scalar('simi', simi_loss, i)
    writer.add_scalar('smooth', smooth_loss, i)
    writer.add_scalar('jdet', jdet_loss, i)
    writer.add_scalar('cyclic', cyclic_loss, i)
    writer.add_scalar('total', total_loss, i)


def write_validation_loss(writer, mean, std, i):
    writer.add_scalar('tre_m', mean, i)
    writer.add_scalar('tre_s', std, i)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


# 构建数据集
def load_data_for_dataset(path):
    # 导入数据集，尺寸: [10, 1, 94, 256, 256]/归一化/裁切
    # image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('.nii')])[0:6]
    # assert image_file_list != None , f'the target file {data_folder} is empty!'
    # image_list = []
    # image_list = [np.flip(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, file_name))), 0) for file_name in
    #               image_file_list]
    # input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
    # input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    # input_image = input_image.squeeze(1)
    image = np.load(path).astype('float32')
    image = torch.from_numpy(image)
    return image


def get_img_path_list(data_folder):
    list_ = []
    for file_name in os.listdir(data_folder):
        list_.append(os.path.join(data_folder, file_name))
    return list_


class SeqDataSet(data.Dataset):
    def __init__(self, data_folder, loader=load_data_for_dataset):
        self.ImgPathList = get_img_path_list(data_folder)
        self.loader = loader

    def __getitem__(self, index):
        ImgPath = self.ImgPathList[index]
        InputImg = self.loader(ImgPath)
        return InputImg

    def __len__(self):
        return len(self.ImgPathList)


class SingleDataSet(data.Dataset):
    def __init__(self, case, loader=load_data_for_dataset):
        self.ImgPathList = [f'/data/JY/Dirlab/case{case}']
        self.loader = loader

    def __getitem__(self, index):
        ImgPath = self.ImgPathList[index]
        InputImg = self.loader(ImgPath)

        return InputImg

    def __len__(self):
        return len(self.ImgPathList)
