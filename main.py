import argparse
import logging
import os

import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] ="3"
import tensorboardX
import torch
import tqdm
from apex import amp

import util
from util import CalTRE, write_loss, load_data, get_case, init_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main(args):
    torch.backends.cudnn.benchmark = True
    case, crop_range, pixel_spacing = get_case(args.case_num)
    # data_folder = f'../DataBase/Dirlab/case{case}/'
    # landmark_file = f'../DataBase/Dirlab/case{case}/Case{case}_300_00_50.pt'
    data_folder = f'/data/JY/Dirlab/case{case}/'
    landmark_file = f'/data/JY/Dirlab/case{case}/Case{case}_300_00_50.pt'
    states_folder = 'result_tmp'

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
    model_name, Model = init_model('ulstm')
    states_file = model_name + f'_case{case}_{index:03d}' if not args.write_name else args.write_name + f'_case{case}_{index:03d}'
    train_writer = tensorboardX.SummaryWriter(os.path.join(states_folder, states_file)) if config.train else None

    # load_data
    input_image, image_shape, num_image = load_data(data_folder, crop_range)
    if config.group_index_list is not None:
        input_image = input_image[config.group_index_list]

    print('input size:', input_image.size())

    # 导入标记点，后续计算TRE
    landmark_info = torch.load(landmark_file)
    landmark_00 = landmark_info['landmark_00']
    landmark_50 = landmark_info['landmark_50']
    landmark_disp = landmark_info['disp_00_50'] if args.group else None

    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    landmark_00_converted = np.flip(landmark_00, axis=1) - np.array(
        [crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype=np.float32)
    landmark_50_converted = np.flip(landmark_50, axis=1) - np.array(
        [crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype=np.float32)

    # 初始化训练模型
    regnet = Model(dim=config.dim, n=len(config.group_index_list), config=config, scale=config.scale)

    regnet = regnet.to(config.device)
    input_image = input_image.to(config.device)
    if args.apex:
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")

    Forward = regnet.pairwise_forward if not args.group else regnet.forward
    Update = regnet.pairwise_update if not args.group else regnet.update
    Sample = regnet.pairwise_sample if not args.group else regnet.sample
    Sample_Slice = regnet.pairwise_sample_slice if not args.group else regnet.sample_slice

    iter = regnet.load() if config.load else 0

    if config.train:
        diff_stats = []
        stop_criterion = util.StopCriterion(stop_std=config.stop_std, query_len=config.stop_query_len)
        pbar = tqdm.tqdm(range(config.max_num_iteration))
        for i in pbar:
            if (i + 1) % 200 == 0:
                Sample_Slice(input_image, os.path.join(states_folder, states_file), i)
                states = {'model': regnet.state_dict()}
                torch.save(states, os.path.join(states_folder, states_file + '.pth'))
                logging.info(f'save model state {states_file} of iter {i}')

            if i % config.pair_disp_calc_interval == 0:
                res = Forward(input_image)
                # 每隔指定轮数，测试TRE
                if args.group:
                    mean, std, diff = regnet.calcdisp.cal_tre(res, config, grid_tuple, landmark_00_converted,
                                                              landmark_disp,
                                                              pixel_spacing)
                else:
                    flow = res['disp_t2i'][config.fixed_disp_indexes]
                    calTRE = CalTRE(grid_tuple, flow)
                    mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)

                diff_stats.append([i, mean, std])
                print(f'\ndiff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f})')

            if args.group:
                simi_loss, smooth_loss, cyclic_loss, total_loss = Update(input_image)
            else:
                simi_loss, smooth_loss, total_loss = Update(input_image)

            stop_criterion.add(simi_loss)
            if stop_criterion.stop():
                break
            pbar.set_description(f'{i + iter}, simi. loss {simi_loss:.4f}, smooth loss {smooth_loss:.3f}')
            write_loss(train_writer, simi_loss, smooth_loss, total_loss, mean, std, i + iter)

        res = Forward(input_image)
        Sample_Slice(input_image, os.path.join(states_folder, states_file), 'best')

        # 训练结束，测试TRE
        if args.group:
            mean, std, diff = regnet.calcdisp.cal_tre(res, config, grid_tuple, landmark_00_converted, landmark_disp,
                                                      pixel_spacing)
        else:
            flow = res['disp_t2i'][config.fixed_disp_indexes]
            calTRE = CalTRE(grid_tuple, flow)
            mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)

        diff_stats.append([i, np.mean(diff), np.std(diff)])
        print(f'\ndiff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f})')

        diff_stats = np.array(diff_stats)

        states = {'model': regnet.state_dict()}
        torch.save(states, os.path.join(states_folder, states_file + '.pth'))

        logging.info(f'save model and optimizer state {states_file}')

    else:
        regnet.eval()
        with torch.no_grad():
            res = Forward(input_image)
            Sample(input_image, config.load[:-4], 'test')
            if args.group:
                mean, std, diff = regnet.calcdisp.cal_tre(res, config, grid_tuple, landmark_00_converted, landmark_disp,
                                                          pixel_spacing)
            else:
                calTRE = CalTRE(grid_tuple, res['disp_t2i'][config.fixed_disp_indexes])
                mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
            if False:
                flow = res['disp_t2i'].detach().cpu()  # .numpy()
                torch.save(res['disp_t2i'], 'result/disp_' + states_file + '.pth')
            print(f'\ndiff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f})')

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('--scale', type=float, default=0.5, help='help')
    parser.add_argument('--lr', type=float, default=1e-2, help='help')
    parser.add_argument('-max', '--max_num_iteration', type=int, default=3000, help='help')
    parser.add_argument('-case', '--case_num', type=int, default=1, help='help')
    parser.add_argument('-channel', '--initial_channels', type=int, default=30, help='help')
    parser.add_argument('--load', type=str, default=None, help='help')
    parser.add_argument('--test', action='store_true', default=False, help='train or test model')
    parser.add_argument('--apex', action='store_true', default=False, help='train or test model')
    parser.add_argument('--group', action='store_true', default=False, help='train methods, groupwise or pairwise')
    parser.add_argument('--model', type=str, default='lstm_se', help='model name')
    parser.add_argument('-name', '--write_name', type=str, default=None, help='name of saved model')
    args = parser.parse_args()
    main(args)
