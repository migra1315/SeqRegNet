import os

import SimpleITK as sitk
import torch
import torch.nn.functional as F
from apex import amp
from torch import nn

import loss
from baseModule import CRNet, UCR


class RegNet(nn.Module):
    """
    生成从增维的T00向图像序列配准的形变场
    """
    def __init__(self, dim=3, seq_len=6, config=None, scale=0.5):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.seq_len = seq_len
        self.scale = scale
        self.config = config

        # self.unet = UlstmCatSkipConnect()
        self.unet = UCR()
        self.spatial_transform = SpatialTransformer(self.dim)

        self.ncc_loss = loss.NCC(self.config['dim'], self.config['ncc_window_size'])
        self.grad = loss.Grad(penalty='l2', loss_mult=2).loss_2D
        self.jac = loss.neg_jdet_loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=1e-3)

    def forward(self, input_image, sample=False):
        if self.config['forward_type'] == 'zero to N':
            return self.forward_zero_to_n(input_image, sample)
        else:
            return self.forward_i_to_i_plus_1(input_image, sample)

    def forward_zero_to_n(self, input_image, sample=False):

        original_image_shape = input_image.shape[2:]
        # input_image: [6, 1, 96, 256, 256]
        # 下采样至1/2
        scaled_image = F.interpolate(input_image, scale_factor=self.scale,
                                     align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                     recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp_t2i = self.unet(scaled_image)

        disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                 mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        # disp_t2i: [6, 3, 96, 256, 256]

        moving_image = input_image[0:1, :, :, :, :].repeat(self.seq_len, 1, 1, 1, 1)

        warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]

        scaled_warped_input_image = F.interpolate(warped_input_image, size=scaled_image_shape,
                                                  mode='bilinear' if self.dim == 2 else 'trilinear',
                                                  align_corners=True)

        # scaled_warped_input_image: [6, 1, 48, 128, 128]
        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image}
        return res

    def forward_i_to_i_plus_1(self, input_image, sample=False):

        original_image_shape = input_image.shape[2:]
        # input_image: [6, 1, 96, 256, 256]
        # 下采样至1/2
        scaled_image = F.interpolate(input_image, scale_factor=self.scale,
                                     align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                     recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp_t2i = self.unet(scaled_image)

        disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                 mode='trilinear', align_corners=True)
        # disp_t2i: [6, 3, 96, 256, 256]

        # moving_image = input_image[0:1, :, :, :, :].repeat(self.seq_len, 1, 1, 1, 1)
        moving_image = torch.cat([input_image[:1], input_image[:-1]], dim=0)
        warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]

        scaled_warped_input_image = F.interpolate(warped_input_image, size=scaled_image_shape,
                                                  mode='bilinear' if self.dim == 2 else 'trilinear',
                                                  align_corners=True)

        # scaled_warped_input_image: [6, 1, 48, 128, 128]
        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image}
        return res

    def update(self, input_image):
        self.optimizer.zero_grad()
        res = self.forward(input_image)
        total_loss = 0.
        simi_loss = self.ncc_loss(res['warped_input_image'], input_image)
        total_loss += simi_loss

        smooth_loss = self.grad(res['scaled_disp_t2i'])
        total_loss += self.config['smooth_reg'] * smooth_loss

        if self.config['jdet_reg'] > 0:
            jdet_loss = self.jac(res['scaled_disp_t2i'])
            total_loss += self.config['jdet_reg'] * jdet_loss
            jdet_loss_item = jdet_loss.item()
        else:
            jdet_loss_item = 0

        if self.config['apex'] is True:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()  # 梯度自动缩放
        else:
            total_loss.backward()
        self.optimizer.step()

        return simi_loss.item(), smooth_loss.item(), jdet_loss_item, 0, total_loss.item()

    def sample(self, input_image, path, iter_, full_sample=False):
        with torch.no_grad():
            self.eval()
            res = self.forward(input_image)
            self.train()
            if full_sample:
                for index in range(6):
                    arr_warped_image = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                    vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
                    vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_warped_image, os.path.join(path, f'warp_{index}_{iter_}.nii'))

                    arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                    vol_disp = sitk.GetImageFromArray(arr_disp)
                    vol_disp.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{index}_{iter_}.nii'))

                    arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                    vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                    vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{index}_{iter_}.nii'))
            else:
                index = -1
                arr_warped_image = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
                vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_warped_image, os.path.join(path, f'warp_{iter_}.nii'))

                arr_disp = res['scaled_disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3,
                                                                                                 0).cpu().numpy()
                vol_disp = sitk.GetImageFromArray(arr_disp)
                vol_disp.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{iter_}.nii'))

                arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{iter_}.nii'))

            arr_moving_image = input_image[0, :, :, :, :].detach().squeeze(0).cpu().numpy()
            vol_moving_image = sitk.GetImageFromArray(arr_moving_image)
            vol_moving_image.SetSpacing([0.976, 0.976, 2.5])
            sitk.WriteImage(vol_moving_image, os.path.join(path, f'moving_{iter_}.nii'))

    def load(self):
        state_file = self.config['load']
        if os.path.exists(state_file):
            states = torch.load(state_file, map_location=self.config['device'])
            #     iter = len(states['loss_list'])
            self.load_state_dict(states['model'])
            print(f'load model and optimizer state {self.config["load"]}')
        else:
            print(f'{self.config["load"]} doesn\'t exist')
            return 0


class RegNet_CR(nn.Module):
    def __init__(self, dim=3, seq_len=6, config=None, scale=0.5):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.seq_len = seq_len
        self.scale = scale
        self.config = config

        self.unet = CRNet()
        self.spatial_transform = SpatialTransformer(self.dim)

        self.ncc_loss = loss.NCC(self.config['dim'], self.config['ncc_window_size'])
        self.grad = loss.Grad(penalty='l2', loss_mult=2).loss_2D
        self.jac = loss.neg_jdet_loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=1e-3)

    def forward(self, input_image, sample=False):

        original_image_shape = input_image.shape[2:]
        scaled_image = F.interpolate(input_image, scale_factor=self.scale,
                                     align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                     recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)
        scaled_image_shape = scaled_image.shape[2:]

        scaled_disp_t2i = self.unet(scaled_image)

        disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                 mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        # disp_t2i: [6, 3, 96, 256, 256]

        moving_image = input_image[0:1, :, :, :, :].repeat(self.seq_len, 1, 1, 1, 1)

        warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]

        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image}
        return res

    def update(self, input_image):
        self.optimizer.zero_grad()
        res = self.forward(input_image)
        total_loss = 0.
        simi_loss = self.ncc_loss(res['warped_input_image'], input_image)
        total_loss += simi_loss

        smooth_loss = self.grad(res['scaled_disp_t2i'])
        total_loss += self.config['smooth_reg'] * smooth_loss
        smooth_loss_item = smooth_loss.item()

        if self.config['jdet_reg'] > 0:
            jdet_loss = self.jac(res['scaled_disp_t2i'])
            total_loss += self.config['jdet_reg'] * jdet_loss
            jdet_loss_item = jdet_loss.item()
        else:
            jdet_loss_item = 0

        if self.config['apex'] is True:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()  # 梯度自动缩放
        else:
            total_loss.backward()
        self.optimizer.step()

        return simi_loss.item(), smooth_loss_item, jdet_loss_item, 0, total_loss.item()

    def sample(self, input_image, path, iter_, full_sample=False):
        with torch.no_grad():
            self.eval()
            res = self.forward(input_image)
            self.train()
            if full_sample:
                for index in range(6):
                    arr_warped_image = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                    vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
                    vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_warped_image, os.path.join(path, f'warp_{index}_{iter_}.nii'))

                    arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                    vol_disp = sitk.GetImageFromArray(arr_disp)
                    vol_disp.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{index}_{iter_}.nii'))

                    arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                    vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                    vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{index}_{iter_}.nii'))
            else:
                index = -1
                arr_warped_image = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
                vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_warped_image, os.path.join(path, f'warp_{iter_}.nii'))

                arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                vol_disp = sitk.GetImageFromArray(arr_disp)
                vol_disp.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{iter_}.nii'))

                arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{iter_}.nii'))

            arr_moving_image = input_image[0, :, :, :, :].detach().squeeze(0).cpu().numpy()
            vol_moving_image = sitk.GetImageFromArray(arr_moving_image)
            vol_moving_image.SetSpacing([0.976, 0.976, 2.5])
            sitk.WriteImage(vol_moving_image, os.path.join(path, f'moving_{iter_}.nii'))

    def load(self):
        state_file = self.config['load']
        if os.path.exists(state_file):
            states = torch.load(state_file, map_location=self.config['device'])
            #     iter = len(states['loss_list'])
            self.load_state_dict(states['model'])
            print(f'load model and optimizer state {self.config["load"]}')
        else:
            print(f'{self.config["load"]} doesn\'t exist')
            return 0


class RegNet_full(nn.Module):
    """
    将两个图像序列同时送入
    """

    def __init__(self, dim=3, seq_len=10, config=None, scale=0.5):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.seq_len = seq_len
        self.scale = scale
        self.config = config

        # self.unet = UlstmCatSkipConnect()
        self.unet = UCR()
        self.spatial_transform = SpatialTransformer(self.dim)

        self.ncc_loss = loss.NCC(self.config['dim'], self.config['ncc_window_size'])
        self.grad = loss.Grad(penalty='l2', loss_mult=2).loss_2D
        self.jac = loss.neg_jdet_loss

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=1e-3)

    def forward(self, input_image, sample=False):
        """
        :param sample: True 返回disp False 返回scale_disp
        :param input_image: [10, 1, 96, 256, 256] 图像顺序 [0, 1, 2, 3, 4, 5, 9, 8, 7, 6]
        :return:
        """
        original_image_shape = input_image.shape[2:]
        scaled_image = F.interpolate(input_image, scale_factor=self.scale, align_corners=True, mode='trilinear',
                                     recompute_scale_factor=False)  # (1, n, d, h, w)
        scaled_image_shape = scaled_image.shape[2:]

        up = scaled_image[[0, 1, 2, 3, 4, 5]]  # 对应T00-T50        0,1,2,3,4,5
        down = scaled_image[[0, 6, 7, 8, 9, 5]]  # 对应T00, T90-T50 0,9,8,7,6,5
        up_disp = self.unet(up)
        down_disp = self.unet(down)
        scaled_disp = torch.cat([up_disp, down_disp[[1, 2, 3, 4]]], dim=0)  # 对应图像时刻 1,2,3,4,5,9,8,7,6,5
        scaled_disp[0] = (up_disp[0] + down_disp[0]) / 2
        scaled_disp[5] = (up_disp[5] + down_disp[5]) / 2
        # print(scaled_disp.size())
        disp = F.interpolate(scaled_disp, size=original_image_shape, mode='trilinear', align_corners=True)

        moving_image = input_image[0:1, :, :, :, :].repeat(10, 1, 1, 1, 1)
        warped_input_image = self.spatial_transform(moving_image, disp)  # 对应图像时刻 0,1,2,3,4,5,9,8,7,6
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]
        if sample:
            res = {'disp_t2i': disp, 'warped_input_image': warped_input_image}
        else:
            res = {'scaled_disp_t2i': scaled_disp, 'warped_input_image': warped_input_image,
                   'disp_diff': (torch.mean((torch.sum((up_disp - down_disp), 0)) ** 2)) ** 0.5}
        return res

    def update(self, input_image):
        self.optimizer.zero_grad()
        res = self.forward(input_image)
        total_loss = 0.
        simi_loss = self.ncc_loss(res['warped_input_image'], input_image)
        total_loss += simi_loss

        # smooth_loss = loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_warped_input_image'])
        smooth_loss = self.grad(res['scaled_disp_t2i'])
        total_loss += self.config['smooth_reg'] * smooth_loss
        smooth_loss_item = smooth_loss.item()

        if self.config['cyc_reg'] > 0:
            cyclic_loss = res['disp_diff']
            total_loss += self.config['cyc_reg'] * cyclic_loss
            cyclic_loss_item = cyclic_loss.item()
        else:
            cyclic_loss_item = 0

        if self.config['jdet_reg'] > 0:
            jdet_loss = self.jac(res['scaled_disp_t2i'])
            total_loss += self.config['jdet_reg'] * jdet_loss
            jdet_loss_item = jdet_loss.item()
        else:
            jdet_loss_item = 0

        if self.config['apex'] is True:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()  # 梯度自动缩放
        else:
            total_loss.backward()
        self.optimizer.step()

        return simi_loss.item(), smooth_loss.item(), jdet_loss_item, cyclic_loss_item, total_loss.item()

    def sample(self, input_image, path, iter_index, full_sample=False):
        with torch.no_grad():
            self.eval()
            res = self.forward(input_image, sample=True)
            self.train()
            if full_sample:
                for index in range(input_image.size()[0]):
                    arr_warped_image = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                    vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
                    vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_warped_image, os.path.join(path, f'warp_{index}_{iter_index}.nii'))

                    arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                    vol_disp = sitk.GetImageFromArray(arr_disp)
                    vol_disp.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{index}_{iter_index}.nii'))

                    arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                    vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                    vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                    sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{index}_{iter_index}.nii'))
            else:
                index = -1
                arr_warped_image = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
                vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_warped_image, os.path.join(path, f'warp_{iter_index}.nii'))

                arr_disp = res['scaled_disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3,
                                                                                                 0).cpu().numpy()
                vol_disp = sitk.GetImageFromArray(arr_disp)
                vol_disp.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{iter_index}.nii'))

                arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{iter_index}.nii'))

            arr_moving_image = input_image[0, :, :, :, :].detach().squeeze(0).cpu().numpy()
            vol_moving_image = sitk.GetImageFromArray(arr_moving_image)
            vol_moving_image.SetSpacing([0.976, 0.976, 2.5])
            sitk.WriteImage(vol_moving_image, os.path.join(path, f'moving_{iter_index}.nii'))

    def load(self):
        state_file = self.config['load']
        if os.path.exists(state_file):
            states = torch.load(state_file, map_location=self.config['device'])
            #     iter = len(states['loss_list'])
            self.load_state_dict(states['model'])
            print(f'load model and optimizer state {state_file}')
        else:
            print(f'{state_file} doesn\'t exist')
            return 0


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, dim, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(dim)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, scale_factor=self.factor, mode=self.mode, align_corners=True,
                              recompute_scale_factor=True)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, scale_factor=self.factor, mode=self.mode, align_corners=True,
                              recompute_scale_factor=True)

        # don't do anything if resize is 1
        return x


class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.grid_dict = {}
        self.norm_coeff_dict = {}

    def forward(self, input_image, flow):
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, 2, h, w) or (n, 3, d, h, w)

        return:
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        img_shape = input_image.shape[2:]
        if img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            grids = torch.meshgrid([torch.arange(0, s) for s in img_shape])
            grid = torch.stack(grids[::-1],
                               dim=0)  # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            grid = torch.unsqueeze(grid, 0)
            grid = grid.to(dtype=flow.dtype, device=flow.device)
            norm_coeff = 2. / (torch.tensor(img_shape[::-1], dtype=flow.dtype,
                                            device=flow.device) - 1.)  # the coefficients to map image coordinates to [-1, 1]
            self.grid_dict[img_shape] = grid
            self.norm_coeff_dict[img_shape] = norm_coeff
            # logging.info(f'\nAdd grid shape {tuple(img_shape)}')
        new_grid = grid + flow

        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1)  # n x h x w x 2
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1)  # n x d x h x w x 3

        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        warped_input_img = F.grid_sample(input_image, new_grid * norm_coeff - 1., mode='bilinear', align_corners=True,
                                         padding_mode='border')
        return warped_input_img
