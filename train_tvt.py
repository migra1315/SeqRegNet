import argparse
import logging
import numpy as np
import os
import random
import shutil
import sys
import tensorboardX
import time
import torch
# from apex import amp
from torch.utils.data import DataLoader

import util

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def train(args_):
    global regnet
    start_time = "{0}-{1}-{2}".format(str(time.localtime(time.time()).tm_mon), str(time.localtime(time.time()).tm_mday),
                                      str(time.localtime(time.time()).tm_hour))
    print(f'start in time {start_time} ... \nloading config file {args_.config} ...')
    config = util.get_config(args_.config)
    if config['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('cuda is not available ...')

    # data_folder = '/data/JY/4_npy_for_seq/'
    data_folder = 'F:\\DataSet\\PKUH3\\4_npy_for_seq'
    states_folder = 'result_tvt'

    os.mkdir(states_folder) if not os.path.exists(states_folder) else None
    index = len([file for file in os.listdir(states_folder) if os.path.isdir(os.path.join(states_folder, file))])
    states_file = config['model'] + f'_{index:03d}'
    train_writer = tensorboardX.SummaryWriter(os.path.join(states_folder, states_file)) if config['train'] else None
    config_save_path = os.path.join(states_folder, states_file, 'config' + start_time + '.yaml')
    shutil.copy(args_.config, config_save_path)

    # load_data
    dataset = util.SeqDataSet(data_folder=data_folder)  # dataset = util.SingleDataSet(case=1)
    loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=8)
    dataset_len = dataset.__len__()

    # 初始化训练模型
    regnet = util.init_model(config['model'])(dim=config['dim'], seq_len=len(config['group_index_list']),
                                              config=config, scale=config['scale']).to(config['device'])

    if config['apex'] is True:
        print('use apex...')
        regnet, regnet.optimizer = amp.initialize(regnet, regnet.optimizer, opt_level="O1")

    for epoch in range(config['max_num_iteration']):
        print(f'epoch {epoch + 1}/{config["max_num_iteration"]}')
        # cal TRE & sample
        if epoch % 200 == 0:
            states = {'model': regnet.state_dict()}
            print(f'---save model of epoch {epoch}---')
            torch.save(states, os.path.join(states_folder, states_file, states_file + f'_{epoch}.pth'))

            with torch.no_grad():
                for case in [1, 3, 5]:
                    if args_.remote:
                        data_folder = f'/data/JY/Dirlab/case{case}/'
                        landmark_file = f'/data/JY/Dirlab/case{case}/Case{case}_300_00_50.pt'
                    else:
                        data_folder = f'../DataBase/Dirlab/case{case}/'
                        landmark_file = f'../DataBase/Dirlab/case{case}/Case{case}_300_00_50.pt'

                    # 导入标记点，后续计算TRE
                    landmark_info = torch.load(landmark_file)
                    landmark_00 = landmark_info['landmark_00']
                    landmark_50 = landmark_info['landmark_50']

                    crop_range, pixel_spacing = util.get_case(case)
                    input_test_image, image_shape, _ = util.load_data_test(data_folder, crop_range)
                    if config['group_index_list'] is not None:
                        input_test_image = input_test_image[config['group_index_list']]
                    input_test_image = input_test_image.to(config['device'])

                    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
                    landmark_00_converted = np.flip(landmark_00, axis=1)
                    landmark_50_converted = np.flip(landmark_50, axis=1)

                    res = regnet.forward(input_test_image)
                    regnet.sample(input_test_image, path=os.path.join(states_folder, states_file),
                                  iter_=f'{epoch}_{case}', full_sample=False)

                    flow = res['disp_t2i'][config['fixed_disp_indexes']]
                    calTRE = util.CalTRE(grid_tuple, flow)
                    mean, std, diff = calTRE.cal_disp(landmark_00_converted, landmark_50_converted, pixel_spacing)
                    print(f'case: {case} of epoch {epoch} diff: {mean:.2f}±{std:.2f}({np.max(diff):.2f})')
                    util.write_validation_loss(train_writer, mean, std, epoch)

        for it, input_image in enumerate(loader):
            # input_image的0维是时间序列，1维是batch
            input_image = input_image.permute(1, 0, 2, 3, 4).to(config['device'])
            # 训练
            simi_loss, smooth_loss, jdet_loss, cyclic_loss, total_loss = regnet.update(input_image)

            print(f'「{it + 1}/{dataset_len} of epoch {epoch + 1}/{config["max_num_iteration"]}」'
                  f', simi. loss {simi_loss:.4f},'
                  f' smooth loss {smooth_loss:.3f},'
                  f' jdet loss {jdet_loss:.3f}'
                  f' cyclic loss {cyclic_loss:.4f}')
            util.write_update_loss(train_writer, simi_loss, smooth_loss,
                                   jdet_loss, cyclic_loss, total_loss, it + epoch * dataset_len)

        if epoch >= config['max_num_iteration']:
            torch.save({'model': regnet.state_dict()},
                       os.path.join(states_folder, states_file, states_file + f'_final.pth'))
            logging.info(f'save model and optimizer state {states_file}')
            sys.exit('Finish training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search some files')
    parser.add_argument('-config', '--config', type=str, default='./config/config_ucr_tvt.yaml')
    parser.add_argument('--remote', action='store_true', default=False, help='train or test model')
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(0)
    train(args)
