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
        sampler_train = DistributedSampler(     ## DistributedSampler : 여러 rank(GPU)간에 데이터를 균등하게 나누고 중복 없이 mini batch로 처리
            opencood_train_dataset,
            num_replicas = opt.world_size,
            rank = opt.rank,                    
            ## Senario 단위 shuffle할지 안할지 결정 [True : Epoch 마다 데이터 순서가 바뀜 ]
            shuffle = True     
        )
        sampler_val = DistributedSampler(
            opencood_val_dataset, 
            num_replicas = opt.world_size,
            rank = opt.rank,
            shuffle = True     ## Senario 단위 shuffle할지 안할지 결정
        ) 

        ## DataLoader num_workers=20이면 최대 num_workders x prefetch_factore(default = 2)만큼 샘플을 미리 준비함
        train_loader = DataLoader(
            opencood_train_dataset,
            # ✅ 핵심: scenario 단위 batch! 
            # -> 현 구조는 Scenario 단위로 묶어서 __getitem__()으로 시퀀스 통째로 반환하기 때문에 이미 batch 개념을 포함하고 있음
            batch_size=1,   
            sampler=sampler_train,
            num_workers=20,     ### num_workers가 커지면, 여러 worker가 같은 파일에 동시 접근하면서 disk I/O 충돌이 날 수 있음
            pin_memory = False,
            collate_fn=opencood_train_dataset.collate_batch
        )

        val_loader = DataLoader(
                opencood_val_dataset,
                batch_size=1,   # ✅ 동일
                sampler=sampler_val,
                num_workers=20,
                pin_memory = False,
                collate_fn=opencood_train_dataset.collate_batch
        )
    
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=20,
                                  collate_fn=opencood_train_dataset.collate_batch,
                                  shuffle=False,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_val_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=20,
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
    num_steps = len(train_loader)               ## 총 scenario 개수 43개
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

    print('Training start with num steps of %d' % num_steps)    ## EX) gpu 2개 사용시, (22, 22)개씩 나누어서 계산한다 여기서 개수가 맞지 않는데 이것은 1개가 duplicated

    num_scenarios_per_epoch = len(train_loader)  # ex) 42

    global_tick_step = 0  # ✅ tick global step

    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:       
            sampler_train.set_epoch(epoch)        ## set_epoch : 매 epoch 마다 seed를 다르게해서, 샘플 순서가 달라지도록 만듬

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        
        ######################################################################################################################
        ######################################################################################################################
        ##                              현재 Dataset을 정리하는게 시간이 조금 걸리는것으로 보임
        ##                              1. DataLoader + collate_batch에서 CPU -> Numpy -> torch 과정이 CPU에서 발생 -> num_workers 조정 필요
        ##                              2. Scenario 단위라 tick수가 길면 메모리가 많이 필요함
        ##                              
        ######################################################################################################################
        ######################################################################################################################
        
        for i,scenario_batch in enumerate(train_loader):
            scenario_id_check = scenario_batch['ego']['scenario_id'][0]
            print(f"[Batch index {i}] scenario_id_check : {scenario_id_check}")
             
            # scenario_batch : dictionary 
            model.train()
            optimizer.zero_grad()
                
            encoding_input_images = scenario_batch['ego']['inputs']         ## ex) INPUT IMAGES :  
            encoding_input_intrins = scenario_batch['ego']['intrinsic']
            encoding_input_extrins = scenario_batch['ego']['extrinsic']
            encoding_input_locs = scenario_batch['ego']['agent_true_loc']

            prev_encoding_result = None
            
            print("INPUT IMAGES : ", encoding_input_images.shape)   
            print("INTRINSIC : ", encoding_input_intrins.shape)   
            print("EXTRINSIC : ", encoding_input_extrins.shape)   
            print("POSITION : ", encoding_input_locs.shape)   
            print("timestamp_key:", scenario_batch['ego']['timestamp_key'].shape)

            assert encoding_input_images.shape[0] == encoding_input_intrins.shape[0] == encoding_input_extrins.shape[0] == encoding_input_locs.shape[0], \
                f"Batch size mismatch: images={encoding_input_images.shape[0]}, intrins={encoding_input_intrins.shape[0]}, extrins={encoding_input_extrins.shape[0]}, locs={encoding_input_locs.shape[0]}"
        
            for tick_idx in range(encoding_input_images.shape[0]):
                
                print(f"tick_idx : {tick_idx} || timestamp_key : {scenario_batch['ego']['timestamp_key'][tick_idx]}", end = ' ')
            print()
                # tick_input_images = encoding_input_images[tick_index].to(device)
                # tick_input_intrins = encoding_input_intrins[tick_index].to(device)
                # tick_input_extrins = encoding_input_extrins[tick_index].to(device)
                # tick_input_locs = encoding_input_locs[tick_index].to(device)
                
                # if not opt.half:
                #     print('This option is not yet set add --half in the end of the commend')
                #     continue
                # else:
                #     with torch.cuda.amp.autocast():
                #         # 첫 tick은 prev_encoding_result 초기화
                #         current_encoding_result = model.module.encoding(
                #             tick_input_images,
                #             tick_input_intrins,
                #             tick_input_extrins,
                #             tick_input_locs,
                #             is_train=True
                #         )

                #         if prev_encoding_result is None:
                #             model_output_dict = model.module.forward(
                #                 current_encoding_result,
                #                 current_encoding_result,  # 첫 tick은 self
                #                 bool_prev_pos_encoded=False
                #             )
                #         else:
                #             model_output_dict = model.module.forward(
                #                 current_encoding_result,
                #                 prev_encoding_result,
                #                 bool_prev_pos_encoded=True
                #             )            
                        
                #         ## 이전에는 tick별 처리로 batch_data로 처리하였지만, 현재는 scenario_batch로 처리하기 떄문에
                #         ## vanila_seg_loss를 위한 임시 dictionary를 생성해줘야함
                #         gt_tick = {
                #             'gt_static' : scenario_batch['ego']['gt_static'][tick_idx].to(device)
                #             'gt_dynamic' : scenario_batch['ego']['gt_dynamic'][tick_idx].to(device)
                #         }
                        
                #         final_loss = criterion(model_output_dict, gt_tick)
                #         # print(final_loss)

        #         prev_encoding_result = current_encoding_result 
                    
        #         if not opt.half:
        #             final_loss.backward(retain_graph=True)
        #             optimizer.step()
        #         else:
        #             scaler.scale(final_loss).backward(retain_graph=True)
        #             scaler.step(optimizer)
        #             scaler.update()
        #         # print('--------------------------------------------------')

        #         scheduler.step_update(global_tick_step)

        #         criterion.logging(epoch, global_tick_step, len(train_loader), writer, pbar=pbar2)
        #         pbar2.update(1)

        #         for lr_idx, param_group in enumerate(optimizer.param_groups):
        #             writer.add_scalar(f'lr_{lr_idx}', param_group["lr"], global_tick_step)

        #         # ✅ tick step 증가!
        #         global_tick_step += 1
                
        # if epoch % hypes['train_params']['eval_freq'] == 0:
        #     valid_ave_loss = []
        #     dynamic_ave_iou = []

        #     for i, scenario_batch in enumerate(val_loader):
        #         model.eval()

        #         scenario_batch = train_utils.to_device(scenario_batch, device)

        #         encoding_input_images = scenario_batch['ego']['inputs']
        #         encoding_input_intrins = scenario_batch['ego']['intrinsic']
        #         encoding_input_extrins = scenario_batch['ego']['extrinsic']
        #         encoding_input_locs = scenario_batch['ego']['agent_true_loc']

        #         prev_encoding_result = None
                
        #         for tick_idx in range(encoding_input_images.shape[0]):
                    
        #             tick_input_images = encoding_input_images[tick_index]
        #             tick_input_intrins = encoding_input_intrins[tick_index]
        #             tick_input_extrins = encoding_input_extrins[tick_index]
        #             tick_input_locs = encoding_input_locs[tick_index]
                    
        #             current_encoding_result = model.module.encoding(
        #             tick_input_images,
        #             tick_input_intrins,
        #             tick_input_extrins,
        #             tick_input_locs,
        #             is_train=False
        #             )

        #             if prev_encoding_result is None:
        #                 model_output_dict = model.module.forward(
        #                     current_encoding_result,
        #                     current_encoding_result,
        #                     bool_prev_pos_encoded=False
        #                 )
        #             else:
        #                 model_output_dict = model.module.forward(
        #                     current_encoding_result,
        #                     prev_encoding_result,
        #                     bool_prev_pos_encoded=True
        #                 )
                    
        #             gt_tick = {
        #                     'gt_static' : scenario_batch['ego']['gt_static'][tick_idx]
        #                     'inputs': scenario_batch['ego']['inputs'][tick_idx],
        #                     'extrinsic': scenario_batch['ego']['extrinsic'][tick_idx],
        #                     'intrinsic': scenario_batch['ego']['intrinsic'][tick_idx],
        #                     'gt_static': scenario_batch['ego']['gt_static'][tick_idx],
        #                     'gt_dynamic': scenario_batch['ego']['gt_dynamic'][tick_idx],
        #                     'transformation_matrix': scenario_batch['ego']['trainsformation_matrix'][tick_idx],
        #                     'pairwise_t_matrix': scenario_batch['ego']['pairwise_t_matrix'][tick_idx],
        #                     'record_len': scenario_batch['ego']['record_len'][tick_idx],
        #                     'scenario_id': scenario_batch['ego']['scenario_id'][tick_idx],
        #                     'agent_true_loc' : scenario_batch['ego']['agent_true_loc'][tick_idx],
        #                     'cav_list' : scenario_batch['ego']['cav_list'][tick_idx],
        #                     # 'dist_to_ego' : distance_all_batch,
        #                     'single_bev' : scenario_batch['ego']['single_bev'][tick_idx],
        #                     'timestamp_key' : scenario_batch['ego']['timestamp_key'][tick_idx]
        #             }
                    
        #             final_loss = criterion(model_output_dict, gt_tick)
        #             valid_ave_loss.append(final_loss.item())

        #             prev_encoding_result = current_encoding_result

        #             # visualization purpose
        #             model_output_dict = \
        #                 opencood_val_dataset.post_process(gt_tick,
        #                                                     model_output_dict)
        #             train_utils.save_bev_seg_binary(model_output_dict,              ## logs/ys_model_xxxxxx/train_vis
        #                                             gt_tick,
        #                                             saved_path,
        #                                             i,
        #                                             epoch)
        #             iou_dynamic = cal_iou_training(gt_tick,
        #                                             model_output_dict)
        #             # static_ave_iou.append(iou_static[1])
        #             dynamic_ave_iou.append(iou_dynamic[1])
        #             # lane_ave_iou.append(iou_static[2])


        #         valid_ave_loss = statistics.mean(valid_ave_loss)
        #         # static_ave_iou = statistics.mean(static_ave_iou)
        #         # lane_ave_iou = statistics.mean(lane_ave_iou)
        #         dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

        #         print('At epoch %d, the validation loss is %f,'
        #             'the dynamic iou is %f, t'
        #             % (epoch,
        #                 valid_ave_loss,
        #                 dynamic_ave_iou,
        #                 ))

        #         writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
        #         writer.add_scalar('Dynamic_Iou', dynamic_ave_iou, epoch)
        #         # writer.add_scalar('Road_IoU', static_ave_iou, epoch)
        #         # writer.add_scalar('Lane_IoU', static_ave_iou, epoch)

        # if epoch % hypes['train_params']['save_freq'] == 0:
        #     torch.save(model_without_ddp.state_dict(),
        #                os.path.join(saved_path,
        #                             'net_epoch%d.pth' % (epoch + 1)))

        # opencood_train_dataset.reinitialize()


if __name__ == '__main__':
    main()