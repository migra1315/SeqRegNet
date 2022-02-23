import argparse
import logging
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] ="3"
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
from apex import amp
from torch.utils.data import DataLoader

import model
import util
from util import CalTRE, write_loss, get_case, SeqDataSet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def train(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    gpu_num = torch.cuda.device_count()

    data_folder = '/data/JY/PKUH3_new/'
    states_folder = 'result_train'

    config = dict(
        train=not args.test, load=args.load, scale=args.scale, max_num_iteration=args.max_num_iteration,
        dim=3, learning_rate=args.lr, apex=args.apex, initial_channels=args.initial_channels, depth=4,
        normalization=True, smooth_reg=1e-3, cyclic_reg=1e-2, ncc_window_size=5, load_optimizer=False,
        group_index_list=[0, 1, 2, 3, 4, 5], fixed_disp_indexes=5, pair_disp_indexes=[0, 5],
        pair_disp_calc_interval=50, stop_std=0.0007, stop_query_len=100,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    config = util.Struct(**config)

    index = len([file for file in os.listdir(states_folder) if os.path.isdir(os.path.join(states_folder, file))])
    states_file = f'ulstm_{index:03d}' if not args.write_name else args.write_name + f'_{index:03d}'
    train_writer = tensorboardX.SummaryWriter(os.path.join(states_folder, states_file)) if config.train else None

    # load_data
    dataset = SeqDataSet(data_folder=data_folder)  # dataset = util.SingleDataSet(case=1)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=8)
    dataset_len = round(dataset.__len__() / gpu_num)

    # 初始化训练模型
    regnet = model.RegNet(dim=config.dim, n=6, config=config, scale=config.scale).to(config.device)
    if args.apex:
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")
    regnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(regnet)
    if gpu_num > 1:
        print("Let's use", gpu_num, "GPUs!")
        regnet = torch.nn.parallel.DistributedDataParallel(regnet, device_ids=[local_rank], output_device=local_rank)

    for epoch in range(config.max_num_iteration):
        for it, input_image in enumerate(loader):
            input_image = input_image.permute(1, 0, 2, 3, 4).to(config.device)
            # 训练
            if args.group:
                simi_loss, smooth_loss, cyclic_loss, total_loss = regnet.module.pairwise_update(input_image)
            else:
                simi_loss, smooth_loss, total_loss = regnet.module.pairwise_update(input_image)
            print(f'{it}/{dataset_len} of epoch {epoch}/{config.max_num_iteration}')
            write_loss(train_writer, simi_loss, smooth_loss, total_loss, 0, 0, it + epoch * dataset_len)

            # cal TRE

        if (epoch + 1) % 20 == 0:
            regnet.module.pairwise_sample_slice(input_image, os.path.join(states_folder, states_file),
                                                f'{it + epoch * dataset_len}')
            regnet.module.pairwise_sample(input_image, os.path.join(states_folder, states_file),
                                          f'{it + epoch * dataset_len}')
            with torch.no_grad():
                diff_stats = []
                for case in [1, 3, 5]:
                    data_folder = f'/data/JY/Dirlab/case{case}/'
                    landmark_file = f'/data/JY/Dirlab/case{case}/Case{case}_300_00_50.pt'
                    case, crop_range, pixel_spacing = get_case(case)
                    input_test_image, image_shape, num_image = util.load_data_test(data_folder, crop_range)
                    if config.group_index_list is not None:
                        input_test_image = input_test_image[config.group_index_list]
                    # 导入标记点，后续计算TRE
                    landmark_info = torch.load(landmark_file)
                    landmark_00 = landmark_info['landmark_00']
                    landmark_50 = landmark_info['landmark_50']
                    landmark_disp = landmark_info['disp_00_50'] if args.group else None

                    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
                    landmark_00_converted = np.flip(landmark_00, axis=1)
                    landmark_50_converted = np.flip(landmark_50, axis=1)
                    input_test_image = input_test_image.to(config.device)
                    res = regnet.pairwise_forward(input_test_image)

                    # 每隔指定轮数，测试TRE
                    if args.group:
                        mean, std, diff = regnet.calcdisp.cal_tre(res, config, grid_tuple, landmark_00_converted,
                                                                  landmark_disp,
                                                                  pixel_spacing)
                    else:
                        flow = res['disp_t2i'][config.fixed_disp_indexes]
                        print(flow.size())
                        calTRE = CalTRE(grid_tuple, flow)
                        mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
                    diff_stats.append([case, mean, std, np.max(diff)])

                for diff_stat in diff_stats:
                    print(f'***case {diff_stat[0]}: {diff_stat[1]:.2f}+-{diff_stat[2]:.2f}({diff_stat[3]:.2f})***')
                    train_writer.add_scalar(f'{diff_stat[0]}', diff_stat[1], it + epoch * dataset_len)

            # save model
        if (epoch + 1) % 20 == 0:
            states = {'model': regnet.module.state_dict()}
            print(f'---save model of epoch {epoch}---')
            torch.save(states, os.path.join(states_folder, states_file + '.pth'))

        if epoch >= config.max_num_iteration:
            states = {'model': regnet.state_dict()}
            torch.save(states, os.path.join(states_folder, states_file + '.pth'))
            regnet.pairwise_sample_slice(input_image, os.path.join(states_folder, states_file), 'final')
            logging.info(f'save model and optimizer state {states_file}')
            sys.exit('Finish training')

    dist.destroy_process_group()

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--scale', type=float, default=0.5, help='help')
    parser.add_argument('--lr', type=float, default=1e-3, help='help')
    parser.add_argument('-max', '--max_num_iteration', type=int, default=1000, help='help')
    parser.add_argument('-case', '--case_num', type=int, default=1, help='help')
    parser.add_argument('-channel', '--initial_channels', type=int, default=30, help='help')
    parser.add_argument('--load', type=str, default=None, help='help')
    parser.add_argument('--test', action='store_true', default=False, help='train or test model')
    parser.add_argument('--apex', action='store_true', default=False, help='train or test model')
    parser.add_argument('--group', action='store_true', default=False, help='train methods, groupwise or pairwise')
    parser.add_argument('--model', type=str, default='lstm_se', help='model name')
    parser.add_argument('-name', '--write_name', type=str, default=None, help='name of saved model')
    args = parser.parse_args()
    train(args)
