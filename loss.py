import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True


class NCC(nn.Module):
    """
    Calculate local normalized cross-correlation coefficient between tow images.

    Parameters
    ----------
    dim : int
        Dimension of the input images.
    windows_size : int
        Side length of the square window to calculate the local NCC.
    """

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


class Grad:
    """
    N-D gradient loss.
    平滑损失
    """

    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss_2D(self, DvF):
        '''
        input:
            DvF:[batch,dim,H,W],如[1,2,256,256]
        method:
            求y方向导数，并存储在矩阵中: DvF[:,0,:-1,:]-DvF[:,0,1:,:]
            相当于dy = u(x+1,y) - u(x,y)
            求x方向导数，并存储在矩阵中: DvF[:,1,:,:-1]-DvF[:,1,:,1:]
            --------
            各个坐标位置的梯度值求平方和(x**2+y**2)
            忽略最后一行/列（分别缺少dx和dy），求均值
        '''
        dx = torch.abs(DvF[:, :, :, :-1] - DvF[:, :, :, 1:])
        dy = torch.abs(DvF[:, :, :-1, :] - DvF[:, :, 1:, :])

        if self.penalty == 'l2':
            dx = torch.mul(dx, dx)
            dy = torch.mul(dy, dy)

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0
        # d = torch.sqrt(dy[:,:,:,:-1]+dx[:,:,:-1,:])
        # grad = torch.mean(d)/2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

    def loss_3D(self, _, y_pred):

        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def smooth_loss(disp, image):
    """
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field.

    Parameters
    ----------
    disp : (n, 2, h, w) or (n, 3, d, h, w)
        displacement field

    image : (n, 1, d, h, w) or (1, 1, d, h, w)

    """

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


def jacboian_det(displacement):
    """
    :param displacement
        displacement:[batch,channels,L,W,D],如[1,3,256,256,96]
        之后permute成1,256,256,96,3

    """
    displacement = displacement.permute(0, 2, 3, 4, 1)

    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


def neg_jdet_loss(displacement):
    selected_neg_jdet = F.relu(-1.0 * jacboian_det(displacement))
    return torch.mean(selected_neg_jdet)


def calculate_jac(flow):
    # flow = res['disp_t2i'].detach()
    jac = jacboian_det(flow)
    exist = (jac < 0) * 1.0
    image_shape = 1
    for size_ in jac.size():
        image_shape = image_shape * size_
    jac_percent = torch.sum(exist) / image_shape * 100
    jac_mean = torch.mean(jac)
    return jac_percent, jac_mean
