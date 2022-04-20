import torch
import torch.nn.functional as F

from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, normalization=True, pooling=False,
                 LeakyReLU_slope=0.2):
        super().__init__()
        block = []
        block.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=bias))
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


class ConvFormer(nn.Module):
    def __init__(self, in_channels, out_channels, position_embedding_size, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(ConvFormer, self).__init__()
        self.conv_q = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.conv_k = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.conv_v = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.attention = Attn(out_channels, out_channels)
        self.conv_forward = ConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        # nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1))
        self.position_embedding = nn.Parameter(torch.zeros(position_embedding_size))

        self.active_relu = nn.ReLU()
        self.active_sigmoid = nn.Sigmoid()

    def forward(self, c, e):
        queries = []
        keys = []
        values = []
        time_length = e.size()[1]
        c = c + self.position_embedding
        e = e + self.position_embedding
        for time_index in range(time_length):
            q = self.conv_q(e[:, time_index, :])
            k = self.conv_k(c[:, time_index, :])
            v = self.conv_v(c[:, time_index, :])
            queries.append(q)
            keys.append(k)
            values.append(v)
        outs = []
        for queries_index in range(time_length):
            out = torch.zeros_like(values[queries_index])
            for keys_index in range(time_length):
                out += self.attention(queries[queries_index], keys[keys_index]) \
                       * values[keys_index]
            out = self.conv_forward(out + queries[queries_index])
            outs.append(out)
        return torch.stack(outs, dim=1)


class Attn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attn, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.active_relu = nn.ReLU()
        self.active_sigmoid = nn.Sigmoid()

    def forward(self, c, p):
        out = self.conv3(self.active_relu(self.conv1(c) + self.conv2(p)))
        scale = self.active_sigmoid(out)
        # out = scale * p + c
        # out = torch.cat((scale * p, c), dim=1)

        return scale


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


class UlstmCatSkipConnect(nn.Module):
    """
    现有实验结果的来源
    """

    def __init__(self):
        super(UlstmCatSkipConnect, self).__init__()
        self.down_conv = ConvBlock(in_channels=1, out_channels=16,
                                   kernel_size=(3, 3, 3), padding=1, bias=True)
        self.down_1 = DownConvLSTMBlock(input_channels=16, hidden_channels=32, kernel_size=(3, 3, 3))
        self.down_2 = DownConvLSTMBlock(input_channels=32, hidden_channels=64, kernel_size=(3, 3, 3))
        self.down_3 = DownConvLSTMBlock(input_channels=64, hidden_channels=128, kernel_size=(3, 3, 3))

        self.up_30 = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=(3, 3, 3))
        self.up_31 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3, 3))

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
        x = torch.stack(embedding_list, dim=1)

        down_list = [x.squeeze(0)]

        x = self.down_1(x)
        down_list.append(x.squeeze(0))  # 32

        x = self.down_2(x)
        down_list.append(x.squeeze(0))  # 64

        x = self.down_3(x)
        x = self.up_30(x)
        x = self.up_31(x)

        x = F.interpolate(x.squeeze(0), size=down_list[2].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        x = torch.cat((x, down_list[2]), dim=1)
        x = self.up_20(x.unsqueeze(0))
        x = self.up_21(x)

        x = F.interpolate(x.squeeze(0), size=down_list[1].size()[2:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)
        x = torch.cat((x, down_list[1]), dim=1)
        x = self.up_10(x.unsqueeze(0))
        x = self.up_11(x)

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
        # 需求input [n, 1, 48, 128, 128]
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
        disp = F.interpolate(disp, size=image_shape, mode='trilinear', align_corners=True, recompute_scale_factor=False)
        return disp


class Ulstm_Conv_Former(nn.Module):
    def __init__(self):
        super(Ulstm_Conv_Former, self).__init__()
        self.down_conv = ConvBlock(in_channels=1, out_channels=16,
                                   kernel_size=(3, 3, 3), padding=1, bias=True)
        self.down_1 = ConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3, 3, 3))
        self.down_2 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3, 3))
        self.down_3 = ConvLSTM(input_dim=64, hidden_dim=128, kernel_size=(3, 3, 3))
        self.up_3 = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=(3, 3, 3))
        self.up_20 = ConvLSTM(input_dim=128, hidden_dim=64, kernel_size=(3, 3, 3))
        self.up_21 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3, 3, 3))
        self.up_10 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3, 3, 3))
        self.up_11 = ConvLSTM(input_dim=32, hidden_dim=16, kernel_size=(3, 3, 3))
        self.up_conv = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.attn_1 = ConvFormer(in_channels=64, out_channels=128, position_embedding_size=[1, 6, 64, 9, 25, 25])
        self.attn_2 = ConvFormer(in_channels=32, out_channels=64, position_embedding_size=[1, 6, 32, 19, 51, 51])

    def forward(self, x):
        embedding_list = []
        # 需求input [n, 1, 48, 128, 128]
        seq_len = x.size()[0]
        image_shape = x.size()[2:]
        for index in range(seq_len):
            embedding = self.down_conv(x[index:index + 1, :])
            embedding_list.append(embedding)
        x = torch.stack(embedding_list, dim=1)
        del embedding_list

        down_list = []
        x = self.down_1(x)
        x = F.interpolate(x.squeeze(0), scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False).unsqueeze(0)
        down_list.append(x)
        x = self.down_2(x)
        x = F.interpolate(x.squeeze(0), scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False).unsqueeze(0)
        down_list.append(x)
        x = self.down_3(x)
        x = F.interpolate(x.squeeze(0), scale_factor=0.5, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False).unsqueeze(0)

        x = self.up_3(x)
        x = F.interpolate(x.squeeze(0), size=down_list[1].size()[3:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False).unsqueeze(0)

        # print("attn1 size:", x.size())
        x = self.attn_1(x, down_list[1])
        x = self.up_21(self.up_20(x))
        x = F.interpolate(x.squeeze(0), size=down_list[0].size()[3:], mode='trilinear', align_corners=True,
                          recompute_scale_factor=False).unsqueeze(0)

        # print("attn2 size:", x.size())
        x = self.attn_2(x, down_list[0])
        x = self.up_11(self.up_10(x))
        x = F.interpolate(x.squeeze(0), scale_factor=2, mode='trilinear', align_corners=True,
                          recompute_scale_factor=False)

        disp_list = []
        for index in range(seq_len):
            disp = self.up_conv(x[index:index + 1, :])
            disp_list.append(disp)
        x = torch.stack(disp_list, dim=1).squeeze(0)
        x = F.interpolate(x, size=image_shape, mode='trilinear', align_corners=True, recompute_scale_factor=False)
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
