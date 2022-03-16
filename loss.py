import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True


class NCC(nn.Module):
    '''
    Calculate local normalized cross-correlation coefficient between tow images.

    Parameters
    ----------
    dim : int
        Dimension of the input images.
    windows_size : int
        Side length of the square window to calculate the local NCC.
    '''

    def __init__(self, dim, windows_size=11):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-4  # numerical stability constant

        self.windows_size = windows_size

        self.pad = windows_size // 2
        self.window_volume = windows_size ** self.dim
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, I, J):
        '''
        Parameters
        ----------
        I and J : (n, 1, h, w) or (n, 1, d, h, w)
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        windows_size : int
            Side length of the square window to calculate the local NCC.

        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient.
        '''
        try:
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ] * self.dim, dtype=I.dtype, device=I.device)
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)

        J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad)
        J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad)
        IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad)

        cross = torch.clamp(IJ_sum - I_sum * J_sum / self.window_volume, min=self.num_stab_const)
        I_var = torch.clamp(I2_sum - I_sum ** 2 / self.window_volume, min=self.num_stab_const)
        J_var = torch.clamp(J2_sum - J_sum ** 2 / self.window_volume, min=self.num_stab_const)

        cc = cross / ((I_var * J_var) ** 0.5)

        return -torch.mean(cc)


def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field.

    Parameters
    ----------
    disp : (n, 2, h, w) or (n, 3, d, h, w)
        displacement field

    image : (n, 1, d, h, w) or (1, 1, d, h, w)

    '''

    image_shape = disp.shape
    dim = len(image_shape[2:])

    d_disp = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype=disp.dtype, device=disp.device)
    d_image = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype=disp.dtype, device=disp.device)

    # forward difference
    if dim == 2:
        d_disp[:, 1, :, :-1, :] = (disp[:, :, 1:, :] - disp[:, :, :-1, :])
        d_disp[:, 0, :, :, :-1] = (disp[:, :, :, 1:] - disp[:, :, :, :-1])
        d_image[:, 1, :, :-1, :] = (image[:, :, 1:, :] - image[:, :, :-1, :])
        d_image[:, 0, :, :, :-1] = (image[:, :, :, 1:] - image[:, :, :, :-1])

    elif dim == 3:
        d_disp[:, 2, :, :-1, :, :] = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        d_disp[:, 1, :, :, :-1, :] = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        d_disp[:, 0, :, :, :, :-1] = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])

        d_image[:, 2, :, :-1, :, :] = (image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        d_image[:, 1, :, :, :-1, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        d_image[:, 0, :, :, :, :-1] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1])

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim=2, keepdims=True) * torch.exp(-torch.abs(d_image)))

    return loss


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

class Get_Ja():

    def loss_2D(self, displacement):
        '''
        input:
            displacement:形变场,[batch,channels,L,W],如[1,2,256,256]
        methods:
            计算形变场的在每个体素处的雅可比行列式,当:
                Jac>1,扩张
                Jac=1,不变
                0<Jac<1,收缩
                Jac<0,折叠
            雅可比矩阵:                                                     雅可比行列式:
            dux/dx = ux(x+1,y)-ux(x,y)  duy/dx = ux(x+1,y)-ux(x,y)          Jac = a11*a22-a12*a21
            dux/dy = ux(x,y+1)-ux(x,y)  duy/dy = ux(x,y+1)-ux(x,y)              = dux/dx*duy/dy-duy/dx*dux/dy
            需要注意的事项:
            为了与物理意义对应,即Jac=1,不变.计算雅可比矩阵时,对角线元素应加1,即:Jac= (dux/dx+1)*(duy/dy+1)-duy/dx*dux/dy
        output:
            D:每个体素处的雅可比行列式,[batch,L-1,W-1],如[1,255,255].参照计算梯度的方法,最后1行/列无梯度,故返回L-1,W-1
        '''
        D_y = (displacement[:, :, 1:, :-1] - displacement[:, :, :-1, :-1])
        D_x = (displacement[:, :, :-1, 1:] - displacement[:, :, :-1, :-1])

        D1 = (D_x[:, 0, :, :] + 1) * (D_y[:, 1, :, :] + 1)
        D2 = D_y[:, 0, :, :] * D_x[:, 1, :, :]
        D = D1 - D2
        return D

    def loss_3D(self, displacement):
        '''
        input:
            displacement:[batch,channels,L,W,D],如[1,3,256,256,96]
            之后permute成1,256,256,96,3
        methods:
            参见2D
        '''
        displacement = displacement.permute(0, 2, 3, 4, 1)

        D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
        D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
        D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

        D = D1 - D2 + D3

        return D
