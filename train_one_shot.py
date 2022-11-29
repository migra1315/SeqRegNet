import argparse
import os
import shutil
import time

import numpy as np
import tensorboardX
import torch
import tqdm
import yaml

import loss
import util
from loss import jacboian_det

from apex import amp


def main(args):
    start_time = "{0}-{1}-{2}".format(str(time.localtime(time.time()).tm_mon), str(time.localtime(time.time()).tm_mday),
                                      str(time.localtime(time.time()).tm_hour))
    print(f'\nstart in time {start_time} ... \nloading config file {args.config} ...')
    config = util.get_config(args.config)
    if config['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('cuda is not available ...')

    print(f'loading image file {args.case_num} with scale {config["scale"]}...')
    case, input_image, pixel_spacing, grid_tuple, landmark_00_converted, landmark_50_converted, image_shape \
        = util.make_ready(args, config)

    print('moving file into specified device...')
    input_image = input_image.to(config['device'])
    print('input size:', input_image.size())

    states_folder = 'result_0627cv'
    os.mkdir(states_folder) if not os.path.exists(states_folder) else None
    index = len([file for file in os.listdir(states_folder) if os.path.isdir(os.path.join(states_folder, file))])

    # 初始化训练模型
    regnet = util.init_model(config['model'])(dim=config['dim'], seq_len=len(config['group_index_list']),
                                              config=config, scale=config['scale']).to(config['device'])
    if config['apex'] is True:
        print('use apex...')
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")

    if config['train']:
        start = time.time()

        states_file = config['model'] + '_' + config['forward_type'] + f'_case{case}_{index:03d}'
        train_writer = tensorboardX.SummaryWriter(os.path.join(states_folder, states_file))
        config_save_path = os.path.join(states_folder, states_file, 'config.yaml')
        shutil.copy(args.config, config_save_path)
        config_to_save = config.copy()

        stop_criterion = util.StopCriterion(stop_std=config['stop_std'], query_len=config['stop_query_len'])
        pbar = tqdm.tqdm(range(config['max_num_iteration']))
        mean_best = float('inf')
        for i in pbar:
            simi_loss, smooth_loss, jdet_loss, cyclic_loss, total_loss = regnet.update(input_image)

            if i % 50 == 0:
                res = regnet.forward(input_image, sample=True)
                # 每隔指定轮数，测试TRE
                cal_tre_func = util.CalTRE_InterFrame(grid_tuple, res['disp_t2i']) \
                    if config['forward_type'] == 'if' \
                    else util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][config['fixed_disp_indexes']])

                mean, std, diff = cal_tre_func.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
                util.write_validation_loss(train_writer, mean, std, i)
                print(f'\ndiff: {mean:.2f}±{std:.2f}({np.max(diff):.2f})')

                jac_percent, jac_mean = loss.calculate_jac(res['disp_t2i'].detach())
                print(f'jacobin percent: {jac_percent:.4f} jacobin Avg.: {jac_mean:.4f}')
                if mean < mean_best:
                    mean_best = mean
                    train_result = {
                        'train': False,
                        'load': os.path.join(states_folder, states_file, 'best.pth'),
                        'tre-best': {'mean': float(format(mean, ".4f")), 'std': float(format(std, ".4f")),
                                     'max': float(format(np.max(diff), ".4f"))},
                        'jac-best': {'percent': float(format(jac_percent, ".4f")),
                                     'mean': float(format(jac_mean, ".4f"))},
                        'iter': i
                    }
                    config_to_save.update(train_result)
                    with open(config_save_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config_to_save, f)

                    states = {'model': regnet.state_dict()}
                    torch.save(states, os.path.join(states_folder, states_file, 'best.pth'))
                    print(f'save model state {states_file} of iter {i}')

            stop_criterion.add(simi_loss)
            if stop_criterion.stop():
                break
            pbar.set_description(f'{i}, simi. loss {simi_loss:.4f},'
                                 f' smooth loss {smooth_loss:.3f},'
                                 f' jdet loss {jdet_loss:.3f}'
                                 f' cyclic loss {cyclic_loss:.4f}')
            util.write_update_loss(train_writer, simi_loss, smooth_loss,
                                   jdet_loss, cyclic_loss, total_loss, i)

        # 训练结束，测试TRE
        end = time.time()
        print('\ntrain time:', end - start)

        res = regnet.forward(input_image, sample=True)
        # regnet.sample(input_image, os.path.join(states_folder, states_file), 'best', full_sample=False)

        cal_tre_func = util.CalTRE_InterFrame(grid_tuple, res['disp_t2i']) \
            if config['forward_type'] == 'if' \
            else util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][config['fixed_disp_indexes']])

        mean, std, diff = cal_tre_func.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
        print(f'diff: {mean:.2f}±{std:.2f}({np.max(diff):.2f})')

        jac_percent, jac_mean = loss.calculate_jac(res['disp_t2i'].detach())
        print(f'jacobin percent: {jac_percent:.4f} %\njacobin Avg.: {jac_mean:.4f}')

        train_result = {
            'tre-final': {'mean': float(format(mean, ".4f")), 'std': float(format(std, ".4f")),
                          'max': float(format(np.max(diff), ".4f"))},
            'jac-final': {'percent': float(format(jac_percent, ".4f")),
                          'mean': float(format(jac_mean, ".4f"))},
            'time': float(format((end - start), ".4f"))
        }
        config_to_save.update(train_result)
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f)

        states = {'model': regnet.state_dict()}
        torch.save(states, os.path.join(states_folder, states_file, 'final.pth'))
        print(f'save model and optimizer state {states_file}')

    else:
        if config['load'] is None:
            raise KeyError('config load shouldn\'t be None when test')
        else:
            regnet.load()
        regnet.eval()
        with torch.no_grad():
            res = regnet.forward(input_image, sample=True)
            # print('disp size:', res['disp_t2i'][config['fixed_disp_indexes']].size())

            # save_path, _ = os.path.split(config['load'])
            # regnet.sample(input_image, save_path, 'test', full_sample=True)

            mean_list = []
            std_list = []
            diff_list = []
            # 300 landmarks
            cal_tre_func = util.CalTRE_InterFrame(grid_tuple, res['disp_t2i']) \
                if config['forward_type'] == 'if' \
                else util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][config['fixed_disp_indexes']])

            mean, std, diff = cal_tre_func.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
            mean_list.append(mean)
            std_list.append(std)
            diff_list.append(diff)

            if False:
                # 75 landmarks
                landmark_dict = util.get_test_landmarks_files(args, config, image_shape)
                cal_tre = util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][1])
                mean, std, diff = cal_tre.cal_disp(landmark_dict['landmark_00'], landmark_dict['landmark_10'],
                                                   pixel_spacing)
                mean_list.append(mean)
                std_list.append(std)
                diff_list.append(diff)

                cal_tre = util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][2])
                mean, std, diff = cal_tre.cal_disp(landmark_dict['landmark_00'], landmark_dict['landmark_20'],
                                                   pixel_spacing)
                mean_list.append(mean)
                std_list.append(std)
                diff_list.append(diff)

                cal_tre = util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][3])
                mean, std, diff = cal_tre.cal_disp(landmark_dict['landmark_00'], landmark_dict['landmark_30'],
                                                   pixel_spacing)
                mean_list.append(mean)
                std_list.append(std)
                diff_list.append(diff)

                cal_tre = util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][4])
                mean, std, diff = cal_tre.cal_disp(landmark_dict['landmark_00'], landmark_dict['landmark_40'],
                                                   pixel_spacing)
                mean_list.append(mean)
                std_list.append(std)
                diff_list.append(diff)

                cal_tre = util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][5])
                mean, std, diff = cal_tre.cal_disp(landmark_dict['landmark_00'], landmark_dict['landmark_50'],
                                                   pixel_spacing)
                mean_list.append(mean)
                std_list.append(std)
                diff_list.append(diff)

                cal_tre = util.CalTRE_Lagrangian(grid_tuple, res['disp_t2i'][0])
                mean, std, diff = cal_tre.cal_disp(landmark_dict['landmark_00'], landmark_dict['landmark_00'],
                                                   pixel_spacing)
                mean_list.append(mean)
                std_list.append(std)
                diff_list.append(diff)

                test_result = {
                    'LGF': {'mean': float(format(mean_list[0], ".4f")),
                            'diff': float(format(np.max(diff_list[0]), ".4f")),
                            'std': float(format(std_list[0], ".4f"))},
                    'INF_1': {'mean': float(format(mean_list[1], ".4f")),
                              'diff': float(format(np.max(diff_list[1]), ".4f")),
                              'std': float(format(std_list[1], ".4f"))},
                    'INF_2': {'mean': float(format(mean_list[2], ".4f")),
                              'diff': float(format(np.max(diff_list[2]), ".4f")),
                              'std': float(format(std_list[2], ".4f"))},
                    'INF_3': {'mean': float(format(mean_list[3], ".4f")),
                              'diff': float(format(np.max(diff_list[3]), ".4f")),
                              'std': float(format(std_list[3], ".4f"))},
                    'INF_4': {'mean': float(format(mean_list[4], ".4f")),
                              'diff': float(format(np.max(diff_list[4]), ".4f")),
                              'std': float(format(std_list[4], ".4f"))},
                    'INF_5': {'mean': float(format(mean_list[5], ".4f")),
                              'diff': float(format(np.max(diff_list[5]), ".4f")),
                              'std': float(format(std_list[5], ".4f"))}
                }
                config_to_save = util.get_config(args.config)
                config_to_save.update(test_result)
                with open(args.config, 'w', encoding='utf-8') as f:
                    yaml.dump(config_to_save, f)
                for index in range(len(mean_list)):
                    print(f'diff: {mean_list[index]:.2f}±{std_list[index]:.2f}({np.max(diff_list[index]):.2f})')
            else:
                print(f'diff: {mean_list[0]:.2f}±{std_list[0]:.2f}({np.max(diff_list[0]):.2f})')

            flow = res['disp_t2i'].detach().cpu()  # .numpy()
            jac = jacboian_det(flow)
            jac_percent, jac_mean = loss.calculate_jac(res['disp_t2i'].detach())
            print(f'jacobin percent: {jac_percent}, jacobin Avg.: {jac_mean} %')
            # for index in range(jac.size()[0]):
            #     jac_array = jac[index, :, :, :].detach().cpu().numpy()
            #     jac_image = sitk.GetImageFromArray(jac_array)
            #     jac_image.SetSpacing([0.976, 0.976, 2.5])
            #     sitk.WriteImage(jac_image, os.path.join(save_path, f'jac_{index}.nii'))


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('-case', '--case_num', type=int, default=8)
    parser.add_argument('-config', '--config', type=str, default='./config/config_fucr.yaml')
    parser.add_argument('--remote', action='store_true', default=False, help='train in remote or local')
    args = parser.parse_args()
    # util.set_random_seed(3407)
    main(args)
    # parser.add_argument('--scale', type=float, default=0.4, help='help')
    # parser.add_argument('--remote', action='store_true', default=False, help='train or test model')
    # parser.add_argument('--lr', type=float, default=1e-2, help='help')
    # parser.add_argument('-max', '--max_num_iteration', type=int, default=3000, help='help')
    # parser.add_argument('-channel', '--initial_channels', type=int, default=30, help='help')
    # parser.add_argument('--load', type=str, default=None, help='help')
    # parser.add_argument('--test', action='store_true', default=False, help='train or test model')
    # parser.add_argument('--apex', action='store_true', default=False, help='train or test model')
    # parser.add_argument('-name', '--write_name', type=str, default=None, help='name of saved model')
    # parser.add_argument('--jac', type=str, default='s', help='name of saved model')
    # parser.add_argument('--model', type=str, default='ucr', help='model name')
