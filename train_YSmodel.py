import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training
from einops import rearrange

## Running command
## python opencood/tools/train_camera.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
## CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env /home/rina/SG/train_YSmodel.py --hypes_yaml /home/rina/SG/opencood/hypes_yaml/opcamera/Test.yaml --half

# ##################### For Checking
# import logging
# import os
# log_dir = './logs'
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'output_train.log')

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,  # 로그 레벨 설정 (INFO 레벨 이상)
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file_path),  # 로그를 파일에 기록
#         logging.StreamHandler()  # 콘솔에 로그 출력
#     ]
# )

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for training')
    opt = parser.parse_args()
    return opt

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Seed Setting----------------------')
    seed = train_utils.init_random_seed(None if opt.seed == 0 else opt.seed)
    hypes['train_params']['seed'] = seed
    print('Set seed to %d' % seed)
    train_utils.set_random_seed(seed)

    print('-----------------Dataset Building------------------')

    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_val_dataset = build_dataset(hypes, visualize=False, train=True,
                                         validate=True)
    
    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset,shuffle=False)
        sampler_val = DistributedSampler(opencood_val_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=30,
                                  collate_fn=opencood_train_dataset.collate_batch)
        val_loader = DataLoader(opencood_val_dataset,
                                sampler=sampler_val,
                                num_workers=30,
                                collate_fn=opencood_train_dataset.collate_batch,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=30,
                                  collate_fn=opencood_train_dataset.collate_batch,
                                  shuffle=False,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_val_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=30,
                                collate_fn=opencood_train_dataset.collate_batch,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    model.to(device)
    model_without_ddp = model
    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # #######################################################################################################################
    # image_encoder = EfficientNetFeatureExtractor().to(device=device)
    # image_encoder.eval()  ## eval mode로 설정
    # bev_generator = BEVGenerator(bev_size=hypes['model']['args']['encoder']['bev_size'], embed_dim=hypes['model']['args']['encoder']['embed_dim'],\
    #                               num_heads=hypes['model']['args']['encoder']['num_heads'], num_cameras=hypes['model']['args']['encoder']['num_cameras']).to(device)
    # #######################################################################################################################

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # lr scheduler setup
    epoches = hypes['train_params']['epoches']
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # ## printing the number of parameters layer by layer
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name:<50} | {param.numel():>8} parameters")

    ### pringing the number of parameters in taotal
    def count_parameters(model):
        return sum(p.numel() for p in model.parmeters() if p.requires_grad)

    # total_params = count_parameters(model)
    # print(f"Total Trainable Parameters: {total_params}")

    print('Training start with num steps of %d' % num_steps)    ## 총 시나리오의 tick 개수\

    process_id = None

    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        ## batch_data => 한 시나리오의 한 tick의 이미지 데이터들
        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)

            if not opt.half:
                print('This option is not yet set add --half in the end of the commend')
                continue
            else:
                with torch.cuda.amp.autocast():
                    # print(f'The number of Agents : {batch_data["ego"]["inputs"].shape[0]}')
    #####################################################################################################################################
    ###                                      
    ###                                      opencodd/data_utils/datasets/__init__.COM_RANGE 를 상당히 높게 조정하여 모든 agent를 포함하게 만들엇음ㄴ
    #####################################################################################################################################
                    encoding_input_images = batch_data['ego']['inputs']
                    # print("??????????????????", encoding_input_images.shape)   ## Torch.Size([2, 4, 512, 512, 3])  !!!! 여기서 이미지는 preprocessing 한 데이터
                    encoding_input_intrins = batch_data['ego']['intrinsic']
                    encoding_input_extrins = batch_data['ego']['extrinsic']
                    encoding_input_locs = batch_data['ego']['agent_true_loc']

                    # print("INPUT IMAGES : ", encoding_input_images.shape)   ## Torch.Size([2, 4, 512, 512, 3])  !!!! 여기서 이미지는 preprocessing 한 데이터
                    # print("INTRINSIC : ", encoding_input_intrins.shape)   ## Torch.Size([2, 4, 3, 3])
                    # print("EXTRINSIC : ", encoding_input_extrins.shape)   ## Torch.Size([2, 4, 3, 3])
                    # print("POSITION : ", encoding_input_locs.shape)   ## Torch.Size([2, 4, 3, 3])

                    if process_id == None:
                        process_id = int(batch_data['ego']['scenario_id'][0][0].item())

                    # print(f'Step : {i} || Processing Scenario ID : {process_id}')
                    # print(f'Current Scenario ID : {int(batch_data["ego"]["scenario_id"][0][0].item())}')

                    ### %%%% senario timestamp index test
                    # print(batch_data['ego']['timestamp_key'])

                    ##################################################################################
                    ###                             dictionary 추가 사항들
                    ##################################################################################
                    # logging.info(f"CAV_List: {batch_data['ego']['cav_list']} ||  Distance : {batch_data['ego']['dist_to_ego']}")  # 여기서 출력되는 값이 로그로 기록됩니다.
                    # print(batch_data['ego']['cav_list'])        ## List 형식 Example) [['650','641','659']]
                    # print(batch_data['ego']['dist_to_ego'])     ## Tensor 형식 Example) tensor([[ 0.0000, 30.7861, 20.9463]], device='cuda:0')
                    # print(batch_data['ego']['scenario_id'])    ## Tensor 형식 Example) tensor([[0., 0., 0.]], device='cuda:0')

                    # print(f"Cav_list at scenario : {batch_data['ego']['scenario_id']}, tick : {i} ||| {batch_data['ego']['cav_list']}")
                    ##################################################################################
                    ##################################################################################

                    ## current encoding result shape : torch.Size([64, 400, 400])
                    if i == 0 or (process_id != int(batch_data['ego']['scenario_id'][0][0].item())):
                        # current_encoding_result, current_corr, current_sigma, current_true_pos, current_selected_cav = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs,is_train=True)      
                        # prev_encoding_result, prev_corr, prev_sigma, prev_true_pos, prev_selected_cav = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs,is_train=True)
                        current_encoding_result = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs, is_train=True)      
                        prev_encoding_result = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs, is_train=True)
                        process_id = int(batch_data['ego']['scenario_id'][0][0].item())
                        # model_output_dict = model.module.forward(current_encoding_result, prev_encoding_result, current_corr, current_sigma, current_true_pos, bool_prev_pos_encoded=False)
                        dummy_corr = 0
                        dummy_sigma = 0
                        dummy_true_pose = torch.rand((1,1,1,1))
                        model_output_dict = model.module.forward(current_encoding_result, prev_encoding_result, dummy_corr, dummy_sigma, dummy_true_pose, bool_prev_pos_encoded=False)
                    else:
                        # current_encoding_result, current_corr, current_sigma, current_true_pos, current_selected_cav = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs,is_train=True)
                        # model_output_dict = model.module.forward(current_encoding_result, prev_encoding_result, current_corr, current_sigma, current_true_pos, bool_prev_pos_encoded=True)             
                        current_encoding_result = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs, is_train=True)
                        dummy_corr = 0
                        dummy_sigma = 0
                        dummy_true_pose = torch.rand((1,1,1,1))
                        model_output_dict = model.module.forward(current_encoding_result, prev_encoding_result, dummy_corr, dummy_sigma, dummy_true_pose, bool_prev_pos_encoded=True)             
                    
                    # if current_selected_cav is not None:
                    #     print("Current Selected CAV : ", batch_data['ego'].keys())

                    final_loss = criterion(model_output_dict, batch_data['ego'])
                    # print(final_loss)

                    prev_encoding_result = current_encoding_result 
                    
                    # print('--------------------------------------------------')

                criterion.logging(epoch, i, len(train_loader), writer,
                                pbar=pbar2)
                pbar2.update(1)

                # update the lr to tensorboard
                for lr_idx, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar('lr_%d' % lr_idx, param_group["lr"],
                                    epoch * num_steps + i)

                if not opt.half:
                    final_loss.backward(retain_graph=True)
                    optimizer.step()
                else:
                    scaler.scale(final_loss).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            dynamic_ave_iou = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    encoding_input_images = batch_data['ego']['inputs']
                    encoding_input_intrins = batch_data['ego']['intrinsic']
                    encoding_input_extrins = batch_data['ego']['extrinsic']
                    encoding_input_locs = batch_data['ego']['agent_true_loc']
                    
                    # print("INPUT IMAGES : ", encoding_input_images.shape)   ## Torch.Size([2, 4, 512, 512, 3])  !!!! 여기서 이미지는 preprocessing 한 데이터
                    # print("INTRINSIC : ", encoding_input_intrins.shape)   ## Torch.Size([2, 4, 3, 3])
                    # print("EXTRINSIC : ", encoding_input_extrins.shape)   ## Torch.Size([2, 4, 3, 3])
                    # print("POSITION : ", encoding_input_locs.shape)   ## Torch.Size([2, 4, 3, 3])

                    if process_id == None:
                        process_id = int(batch_data['ego']['scenario_id'][0][0].item())

                    # print(f'Processing Scenario ID : {process_id}')
                    # print(f'Current Scenario ID : {int(batch_data["ego"]["scenario_id"][0][0].item())}')

                    ## current encoding result shape : torch.Size([64, 400, 400])
                    if i == 0 or (process_id != int(batch_data['ego']['scenario_id'][0][0].item())):
                        current_encoding_result = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs, is_train=True)      
                        prev_encoding_result= model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs, is_train=True)
                        process_id = int(batch_data['ego']['scenario_id'][0][0].item())
                        dummy_corr = 0
                        dummy_sigma = 0
                        dummy_true_pose = torch.rand((1,1,1,1))
                        model_output_dict = model.module.forward(current_encoding_result, prev_encoding_result, dummy_corr, dummy_sigma, dummy_true_pose, bool_prev_pos_encoded=False)
                    else:
                        dummy_corr = 0
                        dummy_sigma = 0
                        dummy_true_pose = torch.rand((1,1,1,1))
                        current_encoding_result = model.module.encoding(encoding_input_images,encoding_input_intrins,encoding_input_extrins,encoding_input_locs, is_train=True)
                        model_output_dict = model.module.forward(current_encoding_result, prev_encoding_result, dummy_corr, dummy_sigma, dummy_true_pose, bool_prev_pos_encoded=True)

                    final_loss = criterion(model_output_dict,
                                           batch_data['ego'])
                    valid_ave_loss.append(final_loss.item())

                    # visualization purpose
                    model_output_dict = \
                        opencood_val_dataset.post_process(batch_data['ego'],
                                                          model_output_dict)
                    train_utils.save_bev_seg_binary(model_output_dict,              ## logs/ys_model_xxxxxx/train_vis
                                                    batch_data,
                                                    saved_path,
                                                    i,
                                                    epoch)
                    iou_dynamic = cal_iou_training(batch_data,
                                                 model_output_dict)
                    # static_ave_iou.append(iou_static[1])
                    dynamic_ave_iou.append(iou_dynamic[1])
                    # lane_ave_iou.append(iou_static[2])


            valid_ave_loss = statistics.mean(valid_ave_loss)
            # static_ave_iou = statistics.mean(static_ave_iou)
            # lane_ave_iou = statistics.mean(lane_ave_iou)
            dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

            print('At epoch %d, the validation loss is %f,'
                  'the dynamic iou is %f, t'
                   % (epoch,
                    valid_ave_loss,
                    dynamic_ave_iou,
                    ))

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            writer.add_scalar('Dynamic_Iou', dynamic_ave_iou, epoch)
            # writer.add_scalar('Road_IoU', static_ave_iou, epoch)
            # writer.add_scalar('Lane_IoU', static_ave_iou, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        opencood_train_dataset.reinitialize()


if __name__ == '__main__':
    main()