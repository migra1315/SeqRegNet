import argparse
import logging
import os
import random
import sys
import torch.multiprocessing as mp

# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=3 --node_rank=0  --master_port=6005 train_tvt.py --apex

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


def train_dist(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    gpu_num = torch.cuda.device_count()

    data_folder = '/data/JY/4_npy_for_seq/'  # data_folder='F:\\DataSet\\PKUH3\\4_npy_for_seq'
    states_folder = 'result_tvt'

    config = dict(
        train=not args.test, load=args.load,  max_num_iteration=args.max_num_iteration,
        scale=args.scale, dim=3, learning_rate=args.lr, apex=args.apex,
        smooth_reg=0.05, cyclic_reg=1e-2, ncc_window_size=5,
        group_index_list=[0, 1, 2, 3, 4, 5], fixed_disp_indexes=5, pair_disp_indexes=[0, 5],
        device=device
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    config = util.Struct(**config)

    index = len([file for file in os.listdir(states_folder) if os.path.isdir(os.path.join(states_folder, file))])
    states_file = f'ulstm_{index:03d}' if not args.write_name else args.write_name + f'_{index:03d}'
    train_writer = tensorboardX.SummaryWriter(os.path.join(states_folder, states_file)) if config.train else None

    # load_data
    dataset = SeqDataSet(data_folder=data_folder)  # dataset = util.SingleDataSet(case=1)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=8)
    loader = DataLoader(dataset, batch_size=1, shuffle=(train_sampler is None),
                        sampler=train_sampler, pin_memory=False)
    dataset_len = round(dataset.__len__() / gpu_num)

    # 初始化训练模型
    regnet = model.RegNet(dim=config.dim, seq_len=6, config=config, scale=config.scale).to(device)
    if args.apex:
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")
    if gpu_num > 1:
        print(gpu_num, "GPUs are Used for Training...")
        regnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(regnet)
        regnet = torch.nn.parallel.DistributedDataParallel(regnet, device_ids=[local_rank], output_device=local_rank)

    for epoch in range(config.max_num_iteration):
        for it, input_image in enumerate(loader):
            # input_image的0维是时间序列，1维是batch
            input_image = input_image.permute(1, 0, 2, 3, 4).to(device)
            # 训练
            if args.group:
                simi_loss, smooth_loss, cyclic_loss, total_loss = regnet.module.pairwise_update(input_image)
            else:
                simi_loss, smooth_loss, total_loss = regnet.module.pairwise_update(input_image)
            if local_rank == 0:
                print(f'{it + 1}/{dataset_len} of epoch {epoch + 1}/{config.max_num_iteration}')
                write_loss(train_writer, simi_loss, smooth_loss, total_loss, 0, 0, it + epoch * dataset_len)

        # cal TRE & sample
        if (epoch % 100 == 0) & (local_rank == 0):
            regnet.module.pairwise_sample_slice(input_image, os.path.join(states_folder, states_file), epoch)
            regnet.module.pairwise_sample(input_image, os.path.join(states_folder, states_file), epoch)
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
                    input_test_image = input_test_image.to(device)
                    res = regnet.module.pairwise_forward(input_test_image)

                    # 每隔指定轮数，测试TRE
                    if args.group:
                        mean, std, diff = regnet.calcdisp.cal_tre(res, config, grid_tuple, landmark_00_converted,
                                                                  landmark_disp,
                                                                  pixel_spacing)
                    else:
                        flow = res['disp_t2i'][config.fixed_disp_indexes]
                        calTRE = CalTRE(grid_tuple, flow)
                        mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
                    diff_stats.append([case, mean, std, np.max(diff)])

                for diff_stat in diff_stats:
                    print(f'***case {diff_stat[0]}: {diff_stat[1]:.2f}+-{diff_stat[2]:.2f}({diff_stat[3]:.2f})***')
                    train_writer.add_scalar(f'{diff_stat[0]}', diff_stat[1], it + epoch * dataset_len)

            # save model

        if (epoch % 200 == 0) & (local_rank == 0):
            states = {'model': regnet.module.state_dict()}
            print(f'---save model of epoch {epoch}---')
            torch.save(states, os.path.join(states_folder, states_file + '.pth'))

        if epoch >= config.max_num_iteration:
            states = {'model': regnet.module.state_dict()}
            torch.save(states, os.path.join(states_folder, states_file + '.pth'))
            regnet.pairwise_sample_slice(input_image, os.path.join(states_folder, states_file), 'final')
            logging.info(f'save model and optimizer state {states_file}')
            sys.exit('Finish training')

    dist.destroy_process_group()

def train(args):

    data_folder = '/data/JY/4_npy_for_seq/'  # data_folder='F:\\DataSet\\PKUH3\\4_npy_for_seq'
    states_folder = 'result_tvt'

    config = dict(
        train=not args.test, load=args.load,  max_num_iteration=args.max_num_iteration,
        scale=args.scale, dim=3, learning_rate=args.lr, apex=args.apex,
        smooth_reg=args.smooth, cyclic_reg=1e-2, ncc_window_size=5,
        group_index_list=[0, 1, 2, 3, 4, 5], fixed_disp_indexes=5, pair_disp_indexes=[0, 5],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    config = util.Struct(**config)
    device = config.device
    index = len([file for file in os.listdir(states_folder) if os.path.isdir(os.path.join(states_folder, file))])
    states_file = f'ulstm_{index:03d}' if not args.write_name else args.write_name + f'_{index:03d}'
    train_writer = tensorboardX.SummaryWriter(os.path.join(states_folder, states_file)) if config.train else None

    # load_data
    dataset = SeqDataSet(data_folder=data_folder)  # dataset = util.SingleDataSet(case=1)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=8)
    dataset_len = dataset.__len__()

    # 初始化训练模型
    regnet = model.RegNet(dim=config.dim, seq_len=6, config=config, scale=config.scale).to(device)
    if args.apex:
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")

    for epoch in range(config.max_num_iteration):
        print(f'epoch {epoch + 1}/{config.max_num_iteration}')
        for it, input_image in enumerate(loader):
            # input_image的0维是时间序列，1维是batch
            input_image = input_image.permute(1, 0, 2, 3, 4).to(device)
            # 训练
            if args.group:
                simi_loss, smooth_loss, cyclic_loss, total_loss = regnet.pairwise_update(input_image)
            else:
                simi_loss, smooth_loss, total_loss = regnet.pairwise_update(input_image)
            # print(f'{it + 1}/{dataset_len} of epoch {epoch + 1}/{config.max_num_iteration}')
            write_loss(train_writer, simi_loss, smooth_loss, total_loss, 0, 0, it + epoch * dataset_len)

        # cal TRE & sample
        if epoch % 200 == 0:
            regnet.pairwise_sample_slice(input_image, os.path.join(states_folder, states_file), epoch)
            regnet.pairwise_sample(input_image, os.path.join(states_folder, states_file), epoch)
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
                    input_test_image = input_test_image.to(device)
                    res = regnet.pairwise_forward(input_test_image)

                    # 每隔指定轮数，测试TRE
                    if args.group:
                        mean, std, diff = regnet.calcdisp.cal_tre(res, config, grid_tuple, landmark_00_converted,
                                                                  landmark_disp,
                                                                  pixel_spacing)
                    else:
                        flow = res['disp_t2i'][config.fixed_disp_indexes]
                        calTRE = CalTRE(grid_tuple, flow)
                        mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
                    diff_stats.append([case, mean, std, np.max(diff)])

                for diff_stat in diff_stats:
                    print(f'***case {diff_stat[0]}: {diff_stat[1]:.2f}+-{diff_stat[2]:.2f}({diff_stat[3]:.2f})***')
                    train_writer.add_scalar(f'{diff_stat[0]}', diff_stat[1], it + epoch * dataset_len)

            # save model

        if epoch % 200 == 0:
            states = {'model': regnet.state_dict()}
            print(f'---save model of epoch {epoch}---')
            torch.save(states, os.path.join(states_folder, states_file + '.pth'))

        if epoch >= config.max_num_iteration:
            states = {'model': regnet.state_dict()}
            torch.save(states, os.path.join(states_folder, states_file + '.pth'))
            regnet.pairwise_sample_slice(input_image, os.path.join(states_folder, states_file), 'final')
            logging.info(f'save model and optimizer state {states_file}')
            sys.exit('Finish training')

def test(args):
    data_folder: str = '/data/JY/4_npy_for_seq/'  # data_folder='F:\\DataSet\\PKUH3\\4_npy_for_seq'
    states_folder = 'result_tvt'

    config = dict(
        train=not args.test, load=args.load, max_num_iteration=args.max_num_iteration,
        scale=args.scale, dim=3, learning_rate=args.lr, apex=args.apex,
        smooth_reg=0.05, cyclic_reg=1e-2, ncc_window_size=5,
        group_index_list=[0, 1, 2, 3, 4, 5], fixed_disp_indexes=5, pair_disp_indexes=[0, 5],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    config = util.Struct(**config)

    index = len([file for file in os.listdir(states_folder) if os.path.isdir(os.path.join(states_folder, file))])
    states_file = f'ulstm_{index:03d}' if not args.write_name else args.write_name + f'_{index:03d}'

    # 初始化训练模型
    regnet = model.RegNet(dim=config.dim, seq_len=6, config=config, scale=config.scale).to(config.device)
    if args.apex:
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")

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
                calTRE = CalTRE(grid_tuple, flow)
                mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
            diff_stats.append([case, mean, std, np.max(diff)])

        for diff_stat in diff_stats:
            print(f'***case {diff_stat[0]}: {diff_stat[1]:.2f}+-{diff_stat[2]:.2f}({diff_stat[3]:.2f})***')
        # save model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--scale', type=float, default=0.5, help='help')
    parser.add_argument('--lr', type=float, default=1e-2, help='help')
    parser.add_argument('--smooth', type=float, default=0.05, help='help')
    parser.add_argument('-max', '--max_num_iteration', type=int, default=2000, help='help')
    parser.add_argument('--load', type=str, default=None, help='help')
    parser.add_argument('--test', action='store_true', default=False, help='train or test model')
    parser.add_argument('--apex', action='store_true', default=False, help='use apex or not')
    parser.add_argument('--group', action='store_true', default=False, help='train methods, groupwise or pairwise')
    parser.add_argument('-name', '--write_name', type=str, default=None, help='name of saved model')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3, 4, 5"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(0)
    train(args) if not args.test else test(args)
