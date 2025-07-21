import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
# from torch.utils.data import SequentialSampler


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
    
    num_workers = 12

    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_val_dataset = build_dataset(hypes, visualize=False, train=True,
                                         validate=True)
    
    """
    Train chunk ex)
    train chunk index list : [{'scenario_id': 0, 'start_tick': 0, 'end_tick': 30}, 
    {'scenario_id': 0, 'start_tick': 30, 'end_tick': 60}, 
    {'scenario_id': 0, 'start_tick': 60, 'end_tick': 90}, 
    {'scenario_id': 0, 'start_tick': 90, 'end_tick': 120}, 
    {'scenario_id': 0, 'start_tick': 120, 'end_tick': 150},
    {'scenario_id': 0, 'start_tick': 150, 'end_tick': 180}, 
    .....
    {'scenario_id': 1, 'start_tick': 0, 'end_tick': 30}, 
    {'scenario_id': 1, 'start_tick': 30, 'end_tick': 60},
    """

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
        
        # sampler_val = SequentialSampler(opencood_val_dataset)     ### 한 시나리오에 모든 tick을 보고 싶다면 SequentialSampler 사용

        ## DataLoader num_workers=20이면 최대 num_workders x prefetch_factore(default = 2)만큼 샘플을 미리 준비함
        train_loader = DataLoader(
            opencood_train_dataset,
            # ✅ 핵심: scenario 단위 batch! 
            # -> 현 구조는 Scenario 단위로 묶어서 __getitem__()으로 시퀀스 통째로 반환하기 때문에 이미 batch 개념을 포함하고 있음
            batch_size=1,   
            sampler=sampler_train,
            num_workers=num_workers,     ### num_workers가 커지면, 여러 worker가 같은 파일에 동시 접근하면서 disk I/O 충돌이 날 수 있음
            pin_memory = True,
            collate_fn=opencood_train_dataset.collate_batch
        )

        val_loader = DataLoader(
                opencood_val_dataset,
                batch_size=1,   # ✅ 동일
                sampler=sampler_val,
                num_workers=num_workers,
                pin_memory = True,
                collate_fn=opencood_train_dataset.collate_batch
        )
    
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=num_workers,
                                  collate_fn=opencood_train_dataset.collate_batch,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_val_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=num_workers,
                                collate_fn=opencood_train_dataset.collate_batch,
                                shuffle=False,
                                pin_memory=True,
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

    num_scenarios_per_epoch = len(train_loader)  # ex) 43

    global_tick_step = 0  # ✅ tick global step

    ######################################################################################################################
    ######################################################################################################################
    ##                              현재 Dataset을 기존 tick단위가 아닌 Scneario 단위로 묶어 각 rank에 ticks 묶음(chunk)을 전달하므로써
    ##                              prev data를 잘 쓸수 있는 구조로 조정을 한 상태, 또한 tick 단위로 loss backward 상태
    ##                              
    ######################################################################################################################
    ######################################################################################################################
    
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:       
            sampler_train.set_epoch(epoch)        ## set_epoch : 매 epoch 마다 seed를 다르게해서, 샘플 순서가 달라지도록 만듬
        if opt.rank == 0:
            pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        
        for i,scenario_batch in enumerate(train_loader):
            scenario_id_check = scenario_batch['ego']['scenario_id'][0]
             
            # scenario_batch : dictionary 
            model.train()
                
            ## collate batch를 거친 후의 값이므로 이미지 input을 예를 들면 [num_ticks, num_agents (ego 포함), 1, H, W C]
            encoding_input_images = scenario_batch['ego']['inputs']         ## ex) INPUT IMAGES :  torch.Size([50, 5, 1, 4, 512, 512, 3])
            encoding_input_intrins = scenario_batch['ego']['intrinsic']     ## INTRINSIC :  torch.Size([50, 5, 1, 4, 3, 3])
            encoding_input_extrins = scenario_batch['ego']['extrinsic']     ## EXTRINSIC :  torch.Size([50, 5, 1, 4, 4, 4])
            encoding_input_locs = scenario_batch['ego']['agent_true_loc']   ## Treu LOC : torch.Size([50, 5, 1, 6])
                                                                            ## timestamp_key: torch.Size([50, 5, 1])
            prev_encoding_result = None
            
            # print(f"Scenario id : {scenario_id_check}")
            # print("INPUT IMAGES : ", encoding_input_images.shape)   
            # print("INTRINSIC : ", encoding_input_intrins.shape)   
            # print("EXTRINSIC : ", encoding_input_extrins.shape) 
            # print("POSITION : ", encoding_input_locs.shape)   
            # print("timestamp_key:", scenario_batch['ego']['timestamp_key'].shape)

            assert encoding_input_images.shape[0] == encoding_input_intrins.shape[0] == encoding_input_extrins.shape[0] == encoding_input_locs.shape[0], \
                f"Batch size mismatch: images={encoding_input_images.shape[0]}, intrins={encoding_input_intrins.shape[0]}, extrins={encoding_input_extrins.shape[0]}, locs={encoding_input_locs.shape[0]}"
        
            accumulation_steps = 5
            
            for tick_idx in range(encoding_input_images.shape[0]):
                # print(f"Scenario id : {scenario_id_check} || tick_idx : {tick_idx} // {encoding_input_images.shape[0]} || timestamp_key : {scenario_batch['ego']['timestamp_key'][tick_idx].squeeze().tolist()[0]}")
                tick_input_images = encoding_input_images[tick_idx].to(device)
                tick_input_intrins = encoding_input_intrins[tick_idx].to(device)
                tick_input_extrins = encoding_input_extrins[tick_idx].to(device)
                tick_input_locs = encoding_input_locs[tick_idx].to(device)
                
                # print(f'Before Padding Remove : {tick_input_images.shape}')
                
                if not opt.half:        ## half가 설정 되어있지 않은 경우 (FP32로 계산)
                    current_encoding_result = model.module.encoding(
                        tick_input_images,
                        tick_input_intrins,
                        tick_input_extrins,
                        tick_input_locs,
                        is_train=True
                    )
                    
                    assert torch.isfinite(current_encoding_result).all(), "NaN found in encoding result!"
                    
                    if prev_encoding_result is None:
                        model_output_dict = model.module.forward(
                            current_encoding_result,
                            current_encoding_result,  # 첫 tick은 self
                            bool_prev_pos_encoded=False
                        )
                    else:
                        model_output_dict = model.module.forward(
                            current_encoding_result,
                            prev_encoding_result,
                            bool_prev_pos_encoded=True
                        )            
                    
                    ## 이전에는 tick별 처리로 batch_data로 처리하였지만, 현재는 scenario_batch로 처리하기 떄문에
                    ## vanila_seg_loss를 위한 임시 dictionary를 생성해줘야함
                    gt_tick = {
                        'gt_static' : scenario_batch['ego']['gt_static'][tick_idx].to(device),
                        'gt_dynamic' : scenario_batch['ego']['gt_dynamic'][tick_idx].to(device)
                    }
                    
                    final_loss = criterion(model_output_dict, gt_tick)
                else:
                    with torch.cuda.amp.autocast():
                        # 첫 tick은 prev_encoding_result 초기화
                        current_encoding_result = model.module.encoding(
                            tick_input_images,
                            tick_input_intrins,
                            tick_input_extrins,
                            tick_input_locs,
                            is_train=True
                        )
                        
                        assert torch.isfinite(current_encoding_result).all(), "NaN found in encoding result!"
                        
                        if prev_encoding_result is None:
                            model_output_dict = model.module.forward(
                                current_encoding_result,
                                current_encoding_result,  # 첫 tick은 self
                                bool_prev_pos_encoded=False
                            )
                        else:
                            model_output_dict = model.module.forward(
                                current_encoding_result,
                                prev_encoding_result,
                                bool_prev_pos_encoded=True
                            )            
                        
                        ## 이전에는 tick별 처리로 batch_data로 처리하였지만, 현재는 scenario_batch로 처리하기 떄문에
                        ## vanila_seg_loss를 위한 임시 dictionary를 생성해줘야함
                        gt_tick = {
                            'gt_static' : scenario_batch['ego']['gt_static'][tick_idx].to(device),
                            'gt_dynamic' : scenario_batch['ego']['gt_dynamic'][tick_idx].to(device)
                        }
                        
                        # print(f"GT STatic Unique : {torch.unique(gt_tick['gt_static'])}")
                        # print(f"GT dynamic Unique : {torch.unique(gt_tick['gt_dynamic'])}")
                        
                        # print("Model output static shape:", model_output_dict['static_seg'].shape)
                        # print("GT static shape:", gt_tick['gt_static'].shape)
                        # print("Static output min/max:", model_output_dict['static_seg'].min(), model_output_dict['static_seg'].max())
                        
                        final_loss = criterion(model_output_dict, gt_tick)
                        final_loss = final_loss/accumulation_steps
                        # print(f"Total Loss : {final_loss}")

                prev_encoding_result = current_encoding_result.detach()
                    
                if not opt.half:
                    final_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                else:
                    scaler.scale(final_loss).backward()

                    if (tick_idx +1) % accumulation_steps ==0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()

                # print('--------------------------------------------------')

            
            if opt.rank == 0:
                criterion.logging(epoch, i, scenario_id_check, len(train_loader), writer, pbar=pbar2,rank = opt.rank)
                pbar2.update(1)

                for lr_idx, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'lr_{lr_idx}', param_group["lr"], global_tick_step)
                    

        """
        현재는 Distributed Sampler를 통해 각 rank에 tick chunks를 나누어서 gpu에 할당해주었음
        Visualization save시 현재는 각 gpu rank가 같은 함수에 접근하여 모든 rank가 같은 파일 이름으로 저장하여
        현 시나리오의 모든 tick을 저장하는것이 아닌 tick의 일부만 저장함. 만약 모든 ticks를 visualization 하고 싶으면
        DistributedSampler가 아닌 SequentialSampler를 사용해야함
        현재는 속도가 느린 관계로 Distributed에 일부만 save하는 것으로함 
        """
        # if epoch % hypes['train_params']['eval_freq'] == 0: 
        #     valid_ave_loss = []
        #     static_ave_iou = []
        #     dynamic_ave_iou = []
            
        #     for i, scenario_batch in enumerate(val_loader):
        #         scenario_id_check = scenario_batch['ego']['scenario_id'][0]
        #         model.eval()

        #         scenario_batch = train_utils.to_device(scenario_batch, device)

        #         encoding_input_images = scenario_batch['ego']['inputs']
        #         encoding_input_intrins = scenario_batch['ego']['intrinsic']
        #         encoding_input_extrins = scenario_batch['ego']['extrinsic']
        #         encoding_input_locs = scenario_batch['ego']['agent_true_loc']

        #         prev_encoding_result = None
                
        #         for tick_idx in range(encoding_input_images.shape[0]):
        #             with torch.cuda.amp.autocast():
        #                 print(f"## Validation ## Scenario id : {scenario_id_check} || i : {i} || tick_idx : {tick_idx} // {encoding_input_images.shape[0]} || timestamp_key : {scenario_batch['ego']['timestamp_key'][tick_idx].squeeze().tolist()[0]}")
        #                 tick_input_images = encoding_input_images[tick_idx].to(device)
        #                 tick_input_intrins = encoding_input_intrins[tick_idx].to(device)
        #                 tick_input_extrins = encoding_input_extrins[tick_idx].to(device)
        #                 tick_input_locs = encoding_input_locs[tick_idx].to(device)
                        
        #                 current_encoding_result = model.module.encoding(
        #                 tick_input_images,
        #                 tick_input_intrins,
        #                 tick_input_extrins,
        #                 tick_input_locs,
        #                 is_train=False
        #                 )

        #                 if prev_encoding_result is None:
        #                     model_output_dict = model.module.forward(
        #                         current_encoding_result,
        #                         current_encoding_result,
        #                         bool_prev_pos_encoded=False
        #                     )
        #                 else:
        #                     model_output_dict = model.module.forward(
        #                         current_encoding_result,
        #                         prev_encoding_result,
        #                         bool_prev_pos_encoded=True
        #                     )
                        
        #                 gt_tick = {
        #                         'inputs': scenario_batch['ego']['inputs'][tick_idx],
        #                         'extrinsic': scenario_batch['ego']['extrinsic'][tick_idx],
        #                         'intrinsic': scenario_batch['ego']['intrinsic'][tick_idx],
        #                         'gt_static': scenario_batch['ego']['gt_static'][tick_idx],
        #                         'gt_dynamic': scenario_batch['ego']['gt_dynamic'][tick_idx],
        #                         'transformation_matrix': scenario_batch['ego']['transformation_matrix'][tick_idx],
        #                         'pairwise_t_matrix': scenario_batch['ego']['pairwise_t_matrix'][tick_idx],
        #                         'record_len': scenario_batch['ego']['record_len'][tick_idx],
        #                         'scenario_id': scenario_batch['ego']['scenario_id'][tick_idx],
        #                         'agent_true_loc' : scenario_batch['ego']['agent_true_loc'][tick_idx],
        #                         'cav_list' : scenario_batch['ego']['cav_list'][tick_idx],
        #                         # 'dist_to_ego' : distance_all_batch,
        #                         'single_bev' : scenario_batch['ego']['single_bev'][tick_idx],
        #                         'timestamp_key' : scenario_batch['ego']['timestamp_key'][tick_idx]
        #                 }
                        
        #                 batch_dict = {
        #                                 'ego': {
        #                                     k: v.unsqueeze(0) if torch.is_tensor(v) and v.ndim >= 1 else v
        #                                     for k, v in gt_tick.items()
        #                                 }
        #                             }       ## torch.Size([1, 1, 1, 512, 512])
                        
        #                 ## model output static seg : torch.Size([1, 1, 3, 512, 512])
        #                 ## model output dynamic seg : torch.Size([1, 1, 2, 512, 512])
                        
        #                 final_loss = criterion(model_output_dict, gt_tick)
        #                 valid_ave_loss.append(final_loss.item())
                        
        #                 model_output_dict = \
        #                     opencood_val_dataset.post_process(gt_tick,
        #                                                         model_output_dict)      ## model output dict만 처리
                            
                        
        #                 ## GT input shape : torch.Size([1, 5, 1, 4, 512, 512, 3])       ## agent_nums = 5 ==> max agents pdding이 됨
        #                 ## GT static map : gt_static shape : torch.Size([1, 1, 1, 512, 512])
        #                 ## model_output_dict input shape (static_map) : torch.Size([1, 512, 512])
                        
        #                 train_utils.save_bev_seg_binary(model_output_dict,              ## logs/ys_model_xxxxxx/train_vis
        #                                                 batch_dict,
        #                                                 saved_path,
        #                                                 i,
        #                                                 tick_idx,
        #                                                 epoch)
        #                 iou_cal = cal_iou_training(batch_dict,
        #                                                 model_output_dict)

        #                 static_ave_iou.append(iou_cal[0])
        #                 dynamic_ave_iou.append(iou_cal[1])
        #                 # lane_ave_iou.append(iou_static[2])
                        
        #             prev_encoding_result = current_encoding_result.detach()

        #     valid_ave_loss = statistics.mean(valid_ave_loss)
        #     static_ave_iou = statistics.mean(static_ave_iou)
        #     # lane_ave_iou = statistics.mean(lane_ave_iou)
        #     dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

        #     print('At epoch %d, the validation loss is %f,'
        #         'the dynamic iou is %f, t'
        #         % (epoch,
        #             valid_ave_loss,
        #             dynamic_ave_iou,
        #             ))

        #     writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
        #     writer.add_scalar('Dynamic_Iou', dynamic_ave_iou, epoch)
        #     writer.add_scalar('Road_IoU', static_ave_iou, epoch)
        #         # writer.add_scalar('Lane_IoU', static_ave_iou, epoch)

        # if epoch % hypes['train_params']['save_freq'] == 0:
        #     torch.save(model_without_ddp.state_dict(),
        #                os.path.join(saved_path,
        #                             'net_epoch%d.pth' % (epoch + 1)))

        # opencood_train_dataset.reinitialize()         ## cav 순서가 바뀌어 ego가 바뀌는 상황이 없으므로 reinitialize해줄 필요없음

if __name__ == '__main__':
    main()