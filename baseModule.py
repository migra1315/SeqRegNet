import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, normalization=True, pooling=False,
                 LeakyReLU_slope=0.2):
        super().__init__()
        block = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           padding=padding, bias=bias)]
        if normalization:
            block.append(nn.InstanceNorm3d(out_channels))
        block.append(nn.LeakyReLU(LeakyReLU_slope))
        if pooling:
            block.append(nn.AvgPool3d(kernel_size=(2, 2, 2)))
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
        input_tensor:
            6-D Tensor either of shape (t, b, c, h, w, d) or (b, t, c, h, w, d)
        hidden_state:
            None.
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


class DownConvLSTMBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.block1 = ConvLSTM(input_channels, hidden_channels, kernel_size)

    def forward(self, x, skip_list=None):
        x = self.block1(x).squeeze(0)
        if skip_list:
            skip_list.append(x)
        x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=True)
        return x.unsqueeze(0)


class UCR(nn.Module):
    """
    按照PPT中画的网络结构来做的
    """

    def __init__(self):
        super(UCR, self).__init__()
        self.down_conv = ConvBlock(in_channels=1, out_channels=16,
                                   kernel_size=(3, 3, 3), padding=1, bias=True)
        self.down_1 = DownConvLSTMBlock(input_channels=16, hidden_channels=32, kernel_size=(3, 3, 3))
        self.down_2 = DownConvLSTMBlock(input_channels=32, hidden_channels=64, kernel_size=(3, 3, 3))

        self.bottle_neck_1 = ConvLSTM(input_dim=64, hidden_dim=128, kernel_size=(3, 3, 3))
        self.bottle_neck_2 = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=(3, 3, 3))

        self.up_20 = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=(3, 3, 3))
        self.up_21 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3, 3, 3))

        self.up_10 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3, 3, 3))
        self.up_11 = ConvLSTM(input_dim=32, hidden_dim=16, kernel_size=(3, 3, 3))

        self.up_conv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.up_conv2 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        # 需求input [6, 1, 48, 128, 128]
        seq_len = x.size()[0]
        image_shape = x.size()[2:]

        embedding_list = []
        for index in range(seq_len):
            embedding = self.down_conv(x[index:index + 1, :])
            embedding_list.append(embedding)
        x = torch.stack(embedding_list, dim=1).squeeze(0)
        # print('embedding out:',x.size())
        down_list = [x]
        x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=True).unsqueeze(0)

        x = self.down_1(x, down_list)
        # print('down1 out:',x.size())
        x = self.down_2(x, down_list)
        # print('down2 out:',x.size())

        x = self.bottle_neck_1(x)
        # print('bottle_neck_1:',x.size())
        x = self.bottle_neck_2(x)
        # print('bottle_neck_2:',x.size())

        x = F.interpolate(x.squeeze(0), size=down_list[2].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        # print('bottle_neck_2_sample:',x.size(),down_list[2].size())
        x = torch.cat((x, down_list[2]), dim=1).unsqueeze(0)
        x = self.up_20(x)
        x = self.up_21(x)

        # print('up2 out:',x.size(),down_list[1].size())
        x = F.interpolate(x.squeeze(0), size=down_list[1].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        x = torch.cat((x, down_list[1]), dim=1)
        x = self.up_10(x.unsqueeze(0))
        x = self.up_11(x)

        # print('up1 out:',x.size(),down_list[0].size())
        x = F.interpolate(x.squeeze(0), size=down_list[0].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        x = torch.cat((x, down_list[0]), dim=1)
        disp_list = []
        for index in range(seq_len):
            disp = self.up_conv1(x[index:index + 1, :])
            disp = self.up_conv2(disp)
            disp_list.append(disp)
        x = torch.stack(disp_list, dim=1).squeeze(0)

        x = F.interpolate(x, size=image_shape, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)

        return x


class CRNet(nn.Module):
    """
    根据Lung CRNet复现的
    """

    def __init__(self):
        super(CRNet, self).__init__()
        self.down_conv_1 = ConvBlock(in_channels=1, out_channels=8,
                                     kernel_size=(3, 3, 3), padding=1, bias=True, pooling=True)
        self.down_conv_2 = ConvBlock(in_channels=8, out_channels=16,
                                     kernel_size=(3, 3, 3), padding=1, bias=True, pooling=True)
        self.CR_list = nn.ModuleList()
        self.CR_list.append(ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(3, 3, 3)))
        self.CR_list.append(ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(3, 3, 3)))
        self.CR_list.append(ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(3, 3, 3)))
        self.CR_list.append(ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(3, 3, 3)))

        self.up_conv = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        # 需求input [6, 1, 48, 128, 128]
        seq_len = x.size()[0]
        image_shape = x.size()[2:]

        embedding_list = []
        for index in range(seq_len):
            embedding = self.down_conv_1(x[index:index + 1, :])
            embedding = self.down_conv_2(embedding)
            embedding_list.append(embedding)
        x = torch.stack(embedding_list, dim=1)
        for block in self.CR_list:
            x = block(x)
        x = x.squeeze(0)
        disp_list = []
        for index in range(seq_len):
            disp = self.up_conv(x[index:index + 1, :])
            disp_list.append(disp)
        x = torch.stack(disp_list, dim=1).squeeze(0)

        x = F.interpolate(x, size=image_shape, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
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
