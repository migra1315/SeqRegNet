import os

import SimpleITK as sitk
import torch
import torch.nn.functional as F
from apex import amp
from torch import nn
from torchvision import utils as vutils

import loss
import util


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, normalization=True, LeakyReLU_slope=0.2):
        super().__init__()
        block = []
        block.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=bias))
        if normalization:
            block.append(nn.InstanceNorm3d(out_channels))
        block.append(nn.LeakyReLU(LeakyReLU_slope))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, normalization=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.normalization = normalization
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias
        self.conv = ConvBlock(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,
                              normalization=self.normalization)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width, depth = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, depth,
                            dtype=self.conv.block[0].weight.dtype, device=self.conv.block[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, depth,
                            dtype=self.conv.block[0].weight.dtype, device=self.conv.block[0].weight.device))


class ConvLSTM(nn.Module):
    """
        Parameters:
            input_dim: Number of channels in input
            hidden_dim: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
            return_all_layers: Return the list of computations for all layers
            Note: Will do same padding.
        Input:
            A tensor of size B, T, C, H, W or T, B, C, H, W
        Output:
            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
        Example:
            >> x = torch.rand((32, 10, 64, 128, 128))
            >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
            >> _, last_states = convlstm(x)
            >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1,
                 batch_first=True, bias=True, normalization=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.normalization = normalization
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          normalization=self.normalization))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            6-D Tensor either of shape (t, b, c, h, w, d) or (b, t, c, h, w, d)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w, d) -> (b, t, c, h, w, d)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)
        b, _, _, h, w, d = input_tensor.size()

        '''
        初始化一个长度为num_layers的list
        '''
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w, d))

        seq_len = input_tensor.size(1)
        h, c = hidden_state[0]
        output_inner = []
        # 逐序列的进行卷积运算
        for t in range(seq_len):
            h, c = self.cell_list[0](input_tensor=input_tensor[:, t, :, :, :, :],
                                     cur_state=[h, c])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Attn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attn, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.active_relu = nn.ReLU()
        self.active_sigmoid = nn.Sigmoid()

    def forward(self, c, p):
        out = self.conv3(self.active_relu(self.conv1(c) + self.conv2(p)))
        scale = self.active_sigmoid(out)
        out = scale * p + c
        return out


class Ulstm(nn.Module):
    def __init__(self):
        super(Ulstm, self).__init__()
        self.down_conv = ConvBlock(in_channels=1, out_channels=16,
                                   kernel_size=(3, 3, 3), padding=1, bias=True)
        self.down_1 = ConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3, 3, 3))
        self.down_2 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3, 3))
        self.down_3 = ConvLSTM(input_dim=64, hidden_dim=128, kernel_size=(3, 3, 3))
        self.up_3 = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=(3, 3, 3))
        self.up_2 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3, 3, 3))
        self.up_1 = ConvLSTM(input_dim=32, hidden_dim=16, kernel_size=(3, 3, 3))
        self.up_conv = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=(3, 3, 3), padding=1)
        self.attn_1 = Attn(in_channels=64, out_channels=64)
        self.attn_2 = Attn(in_channels=32, out_channels=32)

    def attn(self, x, y):
        return (x + y) / 2

    def forward(self, x):
        embedding_list = []
        # 需求input [10, 1, 48, 128, 128]
        seq_len = x.size()[0]
        image_shape = x.size()[2:]
        for index in range(seq_len):
            embedding = self.down_conv(x[index:index + 1, :])
            embedding_list.append(embedding)
        embedding = torch.stack(embedding_list, dim=1)
        del embedding_list

        down_list = []
        x = self.down_1(embedding)
        del embedding
        x = F.interpolate(x.squeeze(0), scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        down_list.append(x)
        x = self.down_2(x.unsqueeze(0))
        x = F.interpolate(x.squeeze(0), scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        down_list.append(x)
        x = self.down_3(x.unsqueeze(0))
        x = F.interpolate(x.squeeze(0), scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)

        x = self.up_3(x.unsqueeze(0))
        x = F.interpolate(x.squeeze(0), size=down_list[1].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)

        x = self.attn_1(x, down_list[1])
        # x = self.attn(x, down_list[1])
        x = self.up_2(x.unsqueeze(0))
        x = F.interpolate(x.squeeze(0), size=down_list[0].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)

        x = self.attn_2(x, down_list[0])
        # x = self.attn(x, down_list[0])
        x = self.up_1(x.unsqueeze(0))
        x = F.interpolate(x.squeeze(0), scale_factor=2, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)

        disp_list = []
        for index in range(seq_len):
            disp = self.up_conv(x[index:index + 1, :])
            disp_list.append(disp)
        disp = torch.stack(disp_list, dim=1).squeeze(0)
        del x
        del disp_list
        disp = F.interpolate(disp, size=image_shape, mode='trilinear', align_corners=True,
                             recompute_scale_factor=False)
        return disp


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


class RegNet(nn.Module):

    def __init__(self, dim=3, n=6, config=None, scale=0.5):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.n = n
        self.scale = scale
        self.config = config
        self.unet = Ulstm()
        self.spatial_transform = SpatialTransformer(self.dim)
        self.ncc_loss = loss.NCC(self.config.dim, self.config.ncc_window_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, eps=1e-3)
        self.calcdisp = util.CalcDisp(dim=self.config.dim, calc_device=config.device)
        # self.scaler = GradScaler()
        self.description()

    def forward(self, input_image):
        original_image_shape = input_image.shape[2:]  # d, h, w
        # input : [6, 1, 96, 256, 256]
        # 下采样至1/2
        if self.scale < 1:
            scaled_image = F.interpolate(input_image, scale_factor=self.scale,
                                         align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                         recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)
        else:
            scaled_image = input_image
        # scaled_image : [6, 1, 48, 128, 128]

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp_t2i = self.unet(scaled_image)
        # scaled_disp_t2i: [6, 3, 48, 128, 128]

        if self.scale < 1:
            disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                     mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        else:
            disp_t2i = scaled_disp_t2i
        # disp_t2i: [6, 3, 96, 256, 256]

        warped_input_image = self.spatial_transform(input_image, disp_t2i)
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]
        template = torch.mean(warped_input_image, 0, keepdim=True)  # [1, 1, 96, 256, 256]

        if self.scale < 1:
            scaled_template = F.interpolate(template, size=scaled_image_shape,
                                            mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        else:
            scaled_template = template
        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image,
               'template': template, 'scaled_template': scaled_template}
        return res

    def update(self, input_image):
        self.optimizer.zero_grad()
        res = self.forward(input_image)
        total_loss = 0.
        if 'disp_i2t' in res:
            simi_loss = (self.ncc_loss(res['warped_input_image'], res['template']) + self.ncc_loss(input_image, res[
                'warped_template'])) / 2.
        else:
            simi_loss = self.ncc_loss(res['warped_input_image'], res['template'])
        total_loss += simi_loss

        if self.config.smooth_reg > 0:
            if 'disp_i2t' in res:
                smooth_loss = (loss.smooth_loss(res['scaled_disp_t2i']) + loss.smooth_loss(res['scaled_disp_i2t'])) / 2.
            else:
                smooth_loss = loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_template'])
            total_loss += self.config.smooth_reg * smooth_loss
            smooth_loss_item = smooth_loss.item()
        else:
            smooth_loss_item = 0

        if self.config.cyclic_reg > 0:
            if 'disp_i2t' in res:
                cyclic_loss = ((torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5 + (
                    torch.mean((torch.sum(res['scaled_disp_i2t'], 0)) ** 2)) ** 0.5) / 2.
            else:
                cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5
            total_loss += self.config.cyclic_reg * cyclic_loss
            cyclic_loss_item = cyclic_loss.item()
        else:
            cyclic_loss_item = 0

        total_loss.backward()
        self.optimizer.step()

        return simi_loss.item(), smooth_loss_item, cyclic_loss_item, total_loss.item()

    def sample(self, input_image, path, iter):
        with torch.no_grad():
            res = self.forward(input_image)
            for index in range(res['warped_input_image'].size()[0]):
                arr = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol = sitk.GetImageFromArray(arr)
                sitk.WriteImage(vol, os.path.join(path, f'warp_{iter}_{index}.nii'))

            for index in range(res['disp_t2i'].size()[0]):  # 10,3,83,157,240
                arr = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                vol = sitk.GetImageFromArray(arr)
                sitk.WriteImage(vol, os.path.join(path, f'disp_{iter}_{index}.nii'))

            arr = res['template'].detach().squeeze(0).squeeze(0).cpu().numpy()
            vol = sitk.GetImageFromArray(arr)
            sitk.WriteImage(vol, os.path.join(path, f'tmp.nii'))

    def sample_slice(self, input_image, path, iter):
        with torch.no_grad():
            res = self.forward(input_image)
            for index in range(res['warped_input_image'].size()[0]):
                arr = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0)  # 83,157,240
                slice = arr[30, :, :]
                vutils.save_image(slice.data, os.path.join(path, f'warp_{iter}_{index}.png'), padding=0, normalize=True)

    def pairwise_forward(self, input_image):

        original_image_shape = input_image.shape[2:]
        # input_image: [6, 1, 96, 256, 256]
        # 下采样至1/2
        if self.scale < 1:
            scaled_image = F.interpolate(input_image, scale_factor=self.scale,
                                         align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                         recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)
        else:
            scaled_image = input_image
        # scaled_image: [6, 1, 48, 128, 128]

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp_t2i = self.unet(scaled_image)
        # scaled_disp_t2i: [6, 3, 48, 128, 128]

        if self.scale < 1:
            disp_t2i = F.interpolate(scaled_disp_t2i, size=original_image_shape,
                                     mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        else:
            disp_t2i = scaled_disp_t2i
        # disp_t2i: [6, 3, 96, 256, 256]

        moving_image = input_image[0:1, :, :, :, :].repeat(self.n, 1, 1, 1, 1)

        warped_input_image = self.spatial_transform(moving_image, disp_t2i)
        # [6, 1, 96, 256, 256] @ [6, 3, 96, 256, 256] = [6, 1, 96, 256, 256]

        if self.scale < 1:
            scaled_warped_input_image = F.interpolate(warped_input_image, size=scaled_image_shape,
                                                      mode='bilinear' if self.dim == 2 else 'trilinear',
                                                      align_corners=True)
        else:
            scaled_warped_input_image = warped_input_image
        # scaled_warped_input_image: [6, 1, 48, 128, 128]

        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image,
               'scaled_warped_input_image': scaled_warped_input_image}
        return res

    def pairwise_update(self, input_image):
        self.optimizer.zero_grad()
        # -----混合精度-----#
        # with autocast():
        #     res = self.pairwise_forward(input_image)
        #     total_loss = 0.
        #     simi_loss = self.ncc_loss(res['warped_input_image'], input_image)
        #     total_loss += simi_loss
        #     if self.config.smooth_reg > 0:
        #         smooth_loss = loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_warped_input_image'])
        #         total_loss += self.config.smooth_reg * smooth_loss
        #         smooth_loss_item = smooth_loss.item()
        #     else:
        #         smooth_loss_item = 0
        #
        # self.scaler.scale(total_loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        # -----混合精度-----#
        res = self.pairwise_forward(input_image)
        total_loss = 0.
        simi_loss = self.ncc_loss(res['warped_input_image'], input_image)
        total_loss += simi_loss
        if self.config.smooth_reg > 0:
            smooth_loss = loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_warped_input_image'])
            total_loss += self.config.smooth_reg * smooth_loss
            smooth_loss_item = smooth_loss.item()
        else:
            smooth_loss_item = 0

        if self.config.apex:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()  # 梯度自动缩放
        else:
            total_loss.backward()
        self.optimizer.step()

        return simi_loss.item(), smooth_loss_item, total_loss.item()

    def pairwise_sample(self, input_image, path, iter):
        with torch.no_grad():
            res = self.pairwise_forward(input_image)
            for index in range(res['warped_input_image'].size()[0]):
                arr = (res['warped_input_image']-input_image)[index, :, :, :, :].detach().squeeze(0).cpu().numpy()
                vol = sitk.GetImageFromArray(arr)
                vol.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol, os.path.join(path, f'warp_{iter}_{index}.nii'))

            for index in range(res['disp_t2i'].size()[0]):  # 10,3,83,157,240
                arr = res['disp_t2i'][index, :, :, :, :].detach().squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                vol = sitk.GetImageFromArray(arr)
                vol.SetSpacing([0.976, 0.976, 2.5])
                sitk.WriteImage(vol, os.path.join(path, f'disp_{iter}_{index}.nii'))

    def pairwise_sample_slice(self, input_image, path, iter):
        with torch.no_grad():
            res = self.pairwise_forward(input_image)
            for index in range(res['warped_input_image'].size()[0]):
                arr = res['warped_input_image'][index, :, :, :, :].detach().squeeze(0)  # 83,157,240
                slice = arr[50, :, :]
                vutils.save_image(slice.data, os.path.join(path, f'warp_{iter}_{index}.png'), padding=0, normalize=True)

    def load(self):
        state_file = self.config.load
        if os.path.exists(state_file):
            states = torch.load(state_file, map_location=self.config.device)
            #     iter = len(states['loss_list'])
            self.load_state_dict(states['model'])
            #     if self.config.load_optimizer:
            #         self.optimizer.load_state_dict(states['optimizer'])
            #         logging.info(f'load model and optimizer state {self.config.load} from iter {iter}')
            #     else:
            #         logging.info(f'load model state {self.config.load} from iter {iter}')
            #     return iter
            # else:
            #     logging.info(f'{state_file}doesn\'t exist')
            return 0

    def description(self):
        train = 'train' if self.config.train else 'test'
        scale = self.config.scale
        max_num_iteration = self.config.max_num_iteration
        load = self.config.load if self.config.load else 'None'
        print('------------------------------------')
        print(f'target: {train}; model: ulstm; scale: {scale}; max_num_iteration: {max_num_iteration}; load: {load}')
        print('------------------------------------')