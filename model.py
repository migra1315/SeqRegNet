import os

import SimpleITK as sitk
import torch
import torch.nn.functional as F
from apex import amp
from torch import nn

import loss
from baseModule import CRNet, UCR, SpatialTransformer


class RegNet(nn.Module):
    """
    生成 增维的T00 向 图像序列T00-T50 配准的形变场
    配准网络采用UCR
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

    # def forward(self, input_image, sample=False):
    #     if self.config['forward_type'] == 'lg':
    #         return self.forward_lg(input_image, sample)
    #     elif self.config['forward_type'] == 'if':
    #         return self.forward_if(input_image, sample)
    #     elif self.config['forward_type'] == 'if2lg':
    #         return self.forward_if2lg(input_image, sample)
    #     else:
    #         raise ValueError('Not known forward type')

    def forward(self, input_image, sample=False):
        """
        生成拉格朗日形变场， 增维的T00 向 图像序列T00-T50 配准的形变场。
        """
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
        if self.config['forward_type'] == 'lg':
            moving_image = input_image[0:1, :, :, :, :].repeat(self.seq_len, 1, 1, 1, 1)
            warped_input_image = self.spatial_transform(moving_image, disp_t2i)
            # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]
        elif self.config['forward_type'] == 'if':
            moving_image = torch.cat([input_image[:1], input_image[:-1]], dim=0)
            warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        elif self.config['forward_type'] == 'if2lg':
            scaled_disp_t2i = self.compose_disp_layer(scaled_disp_t2i)
            moving_image = input_image[0:1, :, :, :, :].repeat(self.seq_len, 1, 1, 1, 1)
            warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        else:
            raise ValueError('Not known forward type')

        # scaled_warped_input_image = F.interpolate(warped_input_image, size=scaled_image_shape,
        #                                           mode='trilinear',
        #                                           align_corners=True)
        # scaled_warped_input_image: [6, 1, 48, 128, 128]

        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image}
        return res

    '''
    def forward_if(self, input_image, sample=False):
        
        # 生成帧间形变场，T00,T00-T40 向 图像序列T00-T50 配准的形变场
       
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

    def forward_if2lg(self, input_image, sample=False):
        
        # 从帧间形变场生成拉格朗日形变场
        
        original_image_shape = input_image.shape[2:]
        # input_image: [6, 1, 96, 256, 256]
        # 下采样至1/2
        scaled_image = F.interpolate(input_image, scale_factor=self.scale, align_corners=True,
                                     mode='trilinear', recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp_t2i = self.unet(scaled_image)
        # INF生成的DVF
        # disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
        #                          mode='trilinear', align_corners=True)
        # moving_image = torch.cat([input_image[:1], input_image[:-1]], dim=0)
        # warped_input_image = self.spatial_transform(moving_image, disp_t2i)

        # LGF生成的DVF
        scaled_disp_t2i = self.compose_disp_layer(scaled_disp_t2i)
        disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                 mode='trilinear', align_corners=True)
        moving_image = input_image[0:1, :, :, :, :].repeat(self.seq_len, 1, 1, 1, 1)
        warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]

        scaled_warped_input_image = F.interpolate(warped_input_image, size=scaled_image_shape,
                                                  mode='trilinear', align_corners=True)

        # scaled_warped_input_image: [6, 1, 48, 128, 128]
        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image}
        return res
    '''

    def compose_disp_layer(self, disp):
        disp_list = [disp[0:1]]
        disp_list.append(self.spatial_transform(disp_list[0], disp[1:2]) + disp[1:2])
        disp_list.append(self.spatial_transform(disp_list[1], disp[1:2]) + disp[1:2])
        disp_list.append(self.spatial_transform(disp_list[2], disp[2:3]) + disp[2:3])
        disp_list.append(self.spatial_transform(disp_list[3], disp[3:4]) + disp[3:4])
        disp_list.append(self.spatial_transform(disp_list[4], disp[4:5]) + disp[4:5])
        return torch.stack(disp_list, 0).squeeze(1)

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
            print('save file into ', path)
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
                print('warped', arr_warped_image.max(), arr_warped_image.min())


                arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3,
                                                                                          0).cpu().numpy()
                vol_disp = sitk.GetImageFromArray(arr_disp)
                vol_disp.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_disp, os.path.join(path, f'disp_{iter_}.nii'))

                arr_fixed_image = input_image[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol_fixed_image = sitk.GetImageFromArray(arr_fixed_image)
                vol_fixed_image.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol_fixed_image, os.path.join(path, f'fixed_{iter_}.nii'))
                print('fixed', arr_fixed_image.max(), arr_fixed_image.min())

            arr_moving_image = input_image[0, :, :, :, :].detach().squeeze(0).cpu().numpy()
            vol_moving_image = sitk.GetImageFromArray(arr_moving_image)
            vol_moving_image.SetSpacing([0.976, 0.976, 2.5])
            sitk.WriteImage(vol_moving_image, os.path.join(path, f'moving_{iter_}.nii'))
            print('moving', arr_moving_image.max(), arr_moving_image.min())

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
        scaled_image = F.interpolate(input_image, scale_factor=self.scale, align_corners=True,
                                     mode='trilinear', recompute_scale_factor=False)
        scaled_image_shape = scaled_image.shape[2:]

        scaled_disp_t2i = self.unet(scaled_image)

        disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                 mode='trilinear', align_corners=True)
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
            print('save file into ', path)
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
        if self.config['forward_type'] == 'if2lg':
            # 按照inter-frame方式推理形变场,合成为Lagrangian形变场配准网络
            up_disp = self.compose_disp_layer(up_disp)
            down_disp = self.compose_disp_layer(down_disp)
            scaled_disp = torch.cat([up_disp, down_disp[[1, 2, 3, 4]]], dim=0)  # 对应图像时刻 1,2,3,4,5,9,8,7,6,5
            scaled_disp[0] = (up_disp[0] + down_disp[0]) / 2
            scaled_disp[5] = (up_disp[5] + down_disp[5]) / 2
            disp = F.interpolate(scaled_disp, size=original_image_shape, mode='trilinear', align_corners=True)

            moving_image = input_image[0:1, :, :, :, :].repeat(10, 1, 1, 1, 1)
            warped_input_image = self.spatial_transform(moving_image, disp)  # 对应图像时刻 0,1,2,3,4,5,9,8,7,6

        elif self.config['forward_type'] == 'if':
            # 按照inter-frame方式推理形变场。
            # 输入: up {0, 1, 2, 3, 4, 5} down {0, 9, 8, 7, 6, 5}
            # 生成: {00, 01, 12, 23, 34, 45} {00, 09, 98, 87, 76, 65}
            scaled_disp = torch.cat([up_disp, down_disp[[1, 2, 3, 4, 5]]], dim=0)  # 对应图像时刻 1,2,3,4,5,9,8,7,6,5
            disp = F.interpolate(scaled_disp, size=original_image_shape, mode='trilinear', align_corners=True)

            up_disp = F.interpolate(up_disp, size=original_image_shape, mode='trilinear', align_corners=True)
            moving_image = torch.cat([input_image[:1], input_image[:5]], dim=0)
            up_warped = self.spatial_transform(moving_image, up_disp)

            down_disp = F.interpolate(down_disp, size=original_image_shape, mode='trilinear', align_corners=True)
            moving_image = torch.cat([input_image[:1], input_image[:1], input_image[6:]], dim=0)
            down_warped = self.spatial_transform(moving_image, down_disp)

            # 对应图像时刻 1,2,3,4,5,9,8,7,6,5
            warped_input_image = torch.cat([up_warped, down_warped[[1, 2, 3, 4]]], dim=0)
            warped_input_image[0] = (up_warped[0] + down_warped[0]) / 2
            warped_input_image[5] = (up_warped[5] + down_warped[5]) / 2

        elif self.config['forward_type'] == 'lg':
            # 直接Lagrangian形变场配准网络
            scaled_disp = torch.cat([up_disp, down_disp[[1, 2, 3, 4]]], dim=0)  # 对应图像时刻 1,2,3,4,5,9,8,7,6,5
            scaled_disp[0] = (up_disp[0] + down_disp[0]) / 2
            scaled_disp[5] = (up_disp[5] + down_disp[5]) / 2

            # ----------
            '''
            save_path, _ = os.path.split(self.config['load'])

            arr_warped_image = F.interpolate(down_disp[5:6], size=original_image_shape, mode='trilinear', align_corners=True)[0].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
            vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
            sitk.WriteImage(vol_warped_image, os.path.join(save_path, 'down.nii'))

            arr_warped_image = F.interpolate(up_disp[5:6], size=original_image_shape, mode='trilinear', align_corners=True)[0].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            vol_warped_image = sitk.GetImageFromArray(arr_warped_image)
            vol_warped_image.SetSpacing([0.976, 0.976, 2.5])
            sitk.WriteImage(vol_warped_image, os.path.join(save_path, 'up.nii'))
            '''
            # ----------

            disp = F.interpolate(scaled_disp, size=original_image_shape, mode='trilinear', align_corners=True)

            moving_image = input_image[0:1, :, :, :, :].repeat(10, 1, 1, 1, 1)
            warped_input_image = self.spatial_transform(moving_image, disp)  # 对应图像时刻 0,1,2,3,4,5,9,8,7,6

        else:
            raise ValueError('Not known forward type')

        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]
        if sample:
            res = {'disp_t2i': disp, 'warped_input_image': warped_input_image}
        else:
            res = {'scaled_disp_t2i': scaled_disp, 'warped_input_image': warped_input_image,
                   # 'disp_diff': (torch.mean((torch.sum(up_disp, 0) - torch.sum(down_disp, 0)) ** 2)) ** 0.5}
                   # 'disp_diff': (torch.mean((torch.sum((up_disp - down_disp), 0)) ** 2)) ** 0.5}
                    'disp_diff': (torch.mean((torch.sum((up_disp[5] - down_disp[5]), 0)) ** 2)) ** 0.5}
        return res


    def compose_disp_layer(self, disp):
        disp_list = [disp[0:1]]
        disp_list.append(self.spatial_transform(disp_list[0], disp[1:2]) + disp[1:2])
        disp_list.append(self.spatial_transform(disp_list[1], disp[1:2]) + disp[1:2])
        disp_list.append(self.spatial_transform(disp_list[2], disp[2:3]) + disp[2:3])
        disp_list.append(self.spatial_transform(disp_list[3], disp[3:4]) + disp[3:4])
        disp_list.append(self.spatial_transform(disp_list[4], disp[4:5]) + disp[4:5])
        return torch.stack(disp_list, 0).squeeze(1)

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
            print('save file into ', path)

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

                arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3,
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


class RegNet_Bir(nn.Module):
    """
    从 0123和345 两个序列分别推理0位置向其他位置的形变场。
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
        original_image_shape = input_image.shape[2:]
        # input_image: [6, 1, 96, 256, 256]
        # 下采样至1/2
        scaled_image = F.interpolate(input_image, scale_factor=self.scale,
                                     align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                     recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)
        scaled_image_shape = scaled_image.shape[2:]

        path_1 = scaled_image[[0, 1, 2, 3]]  # 对应T00-T30        0,1,2,3
        path_2 = scaled_image[[3, 4, 5]]  # 对应T3-T50        3, 4, 5
        disp_1 = self.unet(path_1)
        disp_2 = self.unet(path_2)
        scaled_disp_t2i = torch.cat([disp_1, disp_2], dim=0)

        disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                 mode='trilinear', align_corners=True)
        # disp_t2i: [6, 3, 96, 256, 256]

        moving_image = input_image[0:1, :, :, :, :].repeat(4, 1, 1, 1, 1)
        warped_image_1 = self.spatial_transform(moving_image, disp_t2i[[0, 1, 2, 3]])

        moving_image = input_image[3:4, :, :, :, :].repeat(3, 1, 1, 1, 1)
        warped_image_2 = self.spatial_transform(moving_image, disp_t2i[[4, 5, 6]])
        warped_input_image = torch.cat([warped_image_1, warped_image_2[[1, 2]]], dim=0)  # 对应图像时刻 1,2,3,4,5,9,8,7,6,5
        warped_input_image[3] = (warped_image_1[3] + warped_image_2[0]) / 2

        scaled_warped_input_image = F.interpolate(warped_input_image, size=scaled_image_shape,
                                                  mode='trilinear', align_corners=True)

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
            print('save file into ', path)

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

                arr_disp = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3,
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
