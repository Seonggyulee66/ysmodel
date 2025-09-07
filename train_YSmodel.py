import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
# from torch.utils.data import SequentialSampler
import math
from collections import deque

import torch.distributed as dist


import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training
from einops import rearrange
import torch, torch.nn.functional as F
import torch
import torch.nn.functional as F
import time

def unwrap_module(m):
    # DDP/DP 래핑이면 .module, 아니면 자기 자신
    return m.module if hasattr(m, "module") else m

@torch.no_grad()
def quick_monitor_v2(logits, target, th=0.3, ignore_index=255):
    # (B,L,...) → (B*L,...)
    if logits.dim() == 5:
        B, L, C, H, W = logits.shape
        logits = logits.reshape(B*L, C, H, W)
        if target.dim() == 4 and target.size(1) == L:
            target = target.reshape(B*L, H, W)
        elif target.dim() == 5 and target.size(2) == 1:
            target = target.reshape(B*L, 1, H, W)

    # 확률
    prob_fg = torch.sigmoid(logits[:,0]) if logits.size(1)==1 else torch.softmax(logits,1)[:,1]

    # 타깃 + 유효마스크
    if target.dim() == 4 and target.size(1) == 1: target = target[:,0]
    target = target.long()
    valid  = (target != ignore_index)
    if valid.sum() == 0:
        return dict(P=0.0, R=0.0, PPR=0.0, p_mean=float(prob_fg.mean()), y_prev=0.0, tp=0, fp=0, fn=0)

    y    = (target==1) & valid
    pred = (prob_fg>=th) & valid

    tp = (pred & y).sum().item()
    fp = (pred & (~y) & valid).sum().item()
    fn = ((~pred) & y).sum().item()
    P  = tp / max(tp+fp, 1)
    R  = tp / max(tp+fn, 1)
    return dict(P=P, R=R,
                PPR=float(pred.sum().item()/valid.sum().item()),
                p_mean=float(prob_fg[valid].mean()),
                y_prev=float(y.sum().item()/valid.sum().item()),
                tp=tp, fp=fp, fn=fn)


@torch.no_grad()
def sweep_best_threshold(logits, target, ths=None, ignore_index=255):
    device = logits.device
    if ths is None:
        th_low  = torch.tensor([0.001,0.002,0.003,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.04], device=device)
        th_mid  = torch.linspace(0.05, 0.50, 10, device=device)
        th_high = torch.linspace(0.55, 0.95,  9, device=device)
        ths = torch.unique(torch.cat([th_low, th_mid, th_high], dim=0))

    best, bestF1 = None, -1.0
    for th in ths:
        m  = quick_monitor_v2(logits, target, float(th), ignore_index=ignore_index)
        P, R = float(m.get("P",0.0)), float(m.get("R",0.0))
        den  = P + R
        F1   = 0.0 if den <= 1e-9 else (2.0 * P * R) / den
        if F1 > bestF1:
            bestF1, best = F1, {"th": float(th), "F1": F1, **m}

    if best is None:
        best = {"th": 0.01, "F1": 0.0, "P": 0.0, "R": 0.0, "PPR": 0.0}

    print(f"best th={best['th']:.3f} F1={best['F1']:.3f} P={best['P']:.3f} R={best['R']:.3f}")
    return best



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
    
    if getattr(opt, "distributed", False):
        torch.cuda.set_device(opt.gpu)

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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        model_without_ddp = model.module   # ✅ 옵티마/셰이퍼 등은 module을 사용
    else:
        model_without_ddp = model
    
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
    n_iter_per_epoch = num_steps * hypes['chunk_size']  # 예: 162 * 5 = 810
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch)

    # ## printing the number of parameters layer by layer
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name:<50} | {param.numel():>8} parameters")

    ### pringing the number of parameters in taotal
    def count_parameters(model):
        return sum(p.numel() for p in model.parmeters() if p.requires_grad)

    # total_params = count_parameters(model)
    # print(f"Total Trainable Parameters: {total_params}")
    
    # def set_bn_eval(m):
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eval()
    #         m.track_running_stats = False
            
    # model.apply(set_bn_eval)
    
    def has_sigmoid_or_softmax(module: torch.nn.Module):
        names = []
        for name, m in module.named_modules():
            if isinstance(m, (torch.nn.Sigmoid, torch.nn.Softmax)):
                names.append(name)
        return names

    print('Training start with num steps of %d' % num_steps)    ## EX) gpu 2개 사용시, (22, 22)개씩 나누어서 계산한다 여기서 개수가 맞지 않는데 이것은 1개가 duplicated

    num_scenarios_per_epoch = len(train_loader)  # ex) 43

    accum_steps = 3                    # 예: tick 6개마다 1 step
    global_micro = 0
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    
    import math, torch.nn as nn

    def try_set_binary_head_prior(module, fg_prob=0.015):
        m = unwrap_module(module)
        # dynamic/static 이름 힌트로 찾아보되, 못 찾으면 마지막 2채널 Conv2d 사용
        target = None
        for name, mod in m.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.out_channels == 2 and ("dynamic" in name.lower()):
                target = mod
        if target is None:
            for name, mod in m.named_modules():
                if isinstance(mod, nn.Conv2d) and mod.out_channels == 2:
                    target = mod
        if target is None:
            print("[InitPrior] 2-channel head not found. skip.")
            return

        with torch.no_grad():
            if target.bias is None:
                target.bias = nn.Parameter(torch.zeros(2, device=target.weight.device, dtype=target.weight.dtype))
            p = max(1e-6, min(1-1e-6, fg_prob))
            target.bias.zero_()
            target.bias[0] = math.log((1-p)/p)   # BG
            target.bias[1] = math.log(p/(1-p))   # FG
            print("[InitPrior] set bias with fg_prior =", p)

    # 모델/옵티마이저 세팅 뒤, 스크래치 학습일 때만 1회:
    if init_epoch == 0:
        try_set_binary_head_prior(model, fg_prob=0.015)
    
    scheduler.step_update(0)      ## 
    
    ######################################################################################################################
    ######################################################################################################################
    ##                              현재 Dataset을 기존 tick단위가 아닌 Scneario 단위로 묶어 각 rank에 ticks 묶음(chunk)을 전달하므로써
    ##                              prev data를 잘 쓸수 있는 구조로 조정을 한 상태, 또한 tick 단위로 loss backward 상태
    ##                              
    ######################################################################################################################
    ######################################################################################################################
    # ─────────────────────────────────────────────────────────────────────────────
# 에폭 밖(루프 시작 전) 권장: 운영 임곗값 초기화
    # 에폭 시작하기 전에 1회
    th_run = 0.02                     # prior 기반 초기값
    prev_F1_run  = None
    best_th_hist = deque(maxlen=15)   # 최근 best_th 추적
    th_frozen    = False              # 안정화되면 고정
    # ─────────────────────────────────────────────────────────────────────────────

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # --- 에폭별 sweep 수집 버퍼 ---
        epoch_logits = []
        epoch_targets = []

        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)
        if opt.rank == 0:
            pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, scenario_batch in enumerate(train_loader):
            core = unwrap_module(model)  # ← 시나리오 루프 시작에 한 번만 두어도 됨
            scenario_id_check = scenario_batch['ego']['scenario_id'][0]
            model.train()

            prev_encoding_result = None
            record_len = scenario_batch['ego']['record_len']

            for tick_idx in range(record_len):
                tick_input_images = scenario_batch['ego']['inputs'][tick_idx].to(device)
                tick_input_intrins = scenario_batch['ego']['intrinsic'][tick_idx].to(device)
                tick_input_extrins = scenario_batch['ego']['extrinsic'][tick_idx].to(device)
                tick_input_locs = scenario_batch['ego']['agent_true_loc'][tick_idx].to(device)

                # Batch=1 가정 → encoding 입력 차원 맞춤
                tick_input_images = tick_input_images.unsqueeze(1)
                tick_input_intrins = tick_input_intrins.unsqueeze(1)
                tick_input_extrins = tick_input_extrins.unsqueeze(1)
                tick_input_locs = tick_input_locs.unsqueeze(1)

                # GT dict 구성 (tick 단위)
                gt_tick = {
                    'gt_static'  : torch.from_numpy(scenario_batch['ego']['gt_static'][tick_idx]).unsqueeze(1).to(device),
                    'gt_dynamic' : torch.from_numpy(scenario_batch['ego']['gt_dynamic'][tick_idx]).unsqueeze(1).to(device)
                }

                if not opt.half:
                    current_encoding_result = core.encoding(
                        tick_input_images, tick_input_intrins, tick_input_extrins, tick_input_locs, is_train=True
                    )
                    assert torch.isfinite(current_encoding_result).all(), "NaN found in encoding result!"

                    if prev_encoding_result is None:
                        model_output_dict = model(
                            current_encoding_result, current_encoding_result, bool_prev_pos_encoded=False
                        )
                    else:
                        model_output_dict = model(
                            current_encoding_result, prev_encoding_result, bool_prev_pos_encoded=True
                        )

                    # ★ 변경: FP32에서도 에폭 버퍼 수집
                    with torch.no_grad():
                        logit_tick = model_output_dict['dynamic_seg'].detach()     # (B,L,C,H,W)
                        B, L, C, H, W = logit_tick.shape
                        logit_tick = logit_tick.reshape(B*L, C, H, W).float().cpu()
                        tgt_tick = gt_tick['gt_dynamic']
                        if tgt_tick.dim() == 5:    # (B,L,1,H,W) → (B,L,H,W)
                            tgt_tick = tgt_tick[:, :, 0]
                        tgt_tick = tgt_tick.reshape(B*L, H, W).long().cpu()

                        epoch_logits.append(logit_tick)
                        epoch_targets.append(tgt_tick)

                    final_loss = criterion(model_output_dict, gt_tick)

                else:
                    with torch.cuda.amp.autocast():
                        current_encoding_result = core.encoding(
                            tick_input_images, tick_input_intrins, tick_input_extrins, tick_input_locs, is_train=True
                        )
                        assert torch.isfinite(current_encoding_result).all(), "NaN found in encoding result!"

                        if prev_encoding_result is None:
                            model_output_dict = model(
                                current_encoding_result, current_encoding_result, bool_prev_pos_encoded=False
                            )
                        else:
                            model_output_dict = model(
                                current_encoding_result, prev_encoding_result, bool_prev_pos_encoded=True
                            )

                        # ★ 변경: AMP에서도 동일하게 에폭 버퍼 수집(프린트 제거)
                        with torch.no_grad():
                            logit_tick = model_output_dict['dynamic_seg'].detach()     # (B,L,C,H,W)
                            B, L, C, H, W = logit_tick.shape
                            logit_tick = logit_tick.reshape(B*L, C, H, W).float().cpu()
                            tgt_tick = gt_tick['gt_dynamic']
                            if tgt_tick.dim() == 5:
                                tgt_tick = tgt_tick[:, :, 0]
                            tgt_tick = tgt_tick.reshape(B*L, H, W).long().cpu()

                            epoch_logits.append(logit_tick)
                            epoch_targets.append(tgt_tick)

                        final_loss = criterion(model_output_dict,gt_tick)

                # ★ 변경: per-tick 모니터링(quick_monitor_v2) 프린트 제거
                # (필요하면 디버그 플래그로 임시 활성화 가능)

                # backward & step
                if opt.half:
                    scaler.scale(final_loss).backward()
                else:
                    final_loss.backward()
                    
                # DDP여도 model.parameters() 써도 OK (wrapper가 넘겨줌)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                prev_encoding_result = current_encoding_result.detach()
                global_micro += 1

                if opt.half:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                global_update += 1
                scheduler.step_update(global_update)
                # scheduler.step()
            
            print(f"[{i} / {len(train_loader)}]  || Senario {scenario_id_check} for {record_len}th tick Loss : {final_loss:.4f}")

        if opt.rank == 0:
            criterion.logging(epoch, i, scenario_id_check, len(train_loader), writer, pbar=pbar2, rank=opt.rank)
            pbar2.update(1)
            for lr_idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'lr_{lr_idx}', param_group["lr"], global_micro)

        # ─────────────────────────────────────────────────────────────
         # ★ 변경: 에폭 말에만 스윕/운영 임곗값/요약 지표 프린트
        with torch.no_grad():
            if len(epoch_logits) > 0:
                all_logits  = torch.cat(epoch_logits,  0)  # (N, C, H, W)
                all_targets = torch.cat(epoch_targets, 0)   # (N, H, W)

                # 1) 현 운영 임곗값(th_run)에서의 지표 (P, R, F1, PPR, p_mean, y_prev)
                run_metrics = quick_monitor_v2(all_logits, all_targets, th=th_run)

                # 2) 에폭별 최적 임곗값 스윕
                best = sweep_best_threshold(all_logits, all_targets)  # dict: th, F1, P, R, PPR, p_mean, y_prev ...
                th_run = 0.7 * th_run + 0.3 * best['th']              # EMA 스무딩

                P = float(run_metrics.get('P', 0.0))
                R = float(run_metrics.get('R', 0.0))
                den = P + R
                F1_run = 0.0 if den <= 0 else (2.0 * P * R) / den

                # 로깅
                if opt.rank == 0:
                    writer.add_scalar('th_run', th_run, epoch)
                    writer.add_scalar('best_th', best['th'], epoch)
                    writer.add_scalar('F1_run', F1_run, epoch)
                    writer.add_scalar('P_run', run_metrics['P'], epoch)
                    writer.add_scalar('R_run', run_metrics['R'], epoch)
                    writer.add_scalar('PPR_run', run_metrics['PPR'], epoch)
                    writer.add_scalar('y_prev', run_metrics['y_prev'], epoch)
                    writer.add_scalar('p_mean_run', run_metrics['p_mean'], epoch)

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
        #     print('-'*100)
        #     print('                 Evaluation Progressing')
        #     print('-'*100)
        #     for i, scenario_batch in enumerate(val_loader):
        #         scenario_id_check = scenario_batch['ego']['scenario_id'][0]
        #         model.eval()

        #         prev_encoding_result = None
        #         record_len = scenario_batch['ego']['record_len']
                
        #         for tick_idx in range(record_len):
        #             with torch.no_grad():
        #                 with torch.cuda.amp.autocast():
        #                     # print(f"## Validation ## Scenario id : {scenario_id_check} || i : {i} || tick_idx : {tick_idx} // {record_len} || timestamp_key : {scenario_batch['ego']['timestamp_key'][tick_idx].squeeze().tolist()[0]}")
        #                     tick_input_images = scenario_batch['ego']['inputs'][tick_idx].to(device)        ## [num_agents, num_cams, H, H, C]
        #                     tick_input_intrins = scenario_batch['ego']['intrinsic'][tick_idx].to(device)    ## [num_agents, num_cams, 3, 3]
        #                     tick_input_extrins = scenario_batch['ego']['extrinsic'][tick_idx].to(device)    ## [num_agents, num_cams, 4, 4]
        #                     tick_input_locs = scenario_batch['ego']['agent_true_loc'][tick_idx].to(device)  ## [num_agents, 6]
                            
        #                     # Batch는 1 고정이므로 Encoding shape에 맞게 unsqueeze
        #                     tick_input_images = tick_input_images.unsqueeze(1)          ## [num_agents, 1, num_cams, H, H, C]
        #                     tick_input_intrins = tick_input_intrins.unsqueeze(1)        ## [num_agents, 1, num_cams, 3, 3]
        #                     tick_input_extrins = tick_input_extrins.unsqueeze(1)        ## [num_agents, 1, num_cams, 4, 4]
        #                     tick_input_locs = tick_input_locs.unsqueeze(1)              ## [num_agents, 1, 6]
                            
        #                     current_encoding_result = model.module.encoding(
        #                     tick_input_images,
        #                     tick_input_intrins,
        #                     tick_input_extrins,
        #                     tick_input_locs,
        #                     is_train=False
        #                     )

        #                     if prev_encoding_result is None:
        #                         model_output_dict = model.module.forward(
        #                             current_encoding_result,
        #                             current_encoding_result,
        #                             bool_prev_pos_encoded=False
        #                         )
        #                     else:
        #                         model_output_dict = model.module.forward(
        #                             current_encoding_result,
        #                             prev_encoding_result,
        #                             bool_prev_pos_encoded=True
        #                         )
                            
        #                     gt_tick = {
        #                             'inputs': (scenario_batch['ego']['inputs'][tick_idx]).unsqueeze(1).to(device),
        #                             'extrinsic': (scenario_batch['ego']['extrinsic'][tick_idx]).unsqueeze(1).to(device),
        #                             'intrinsic': (scenario_batch['ego']['intrinsic'][tick_idx]).unsqueeze(1).to(device),
        #                             'gt_static': torch.from_numpy(scenario_batch['ego']['gt_static'][tick_idx]).unsqueeze(1).to(device),
        #                             'gt_dynamic': torch.from_numpy(scenario_batch['ego']['gt_dynamic'][tick_idx]).unsqueeze(1).to(device),
        #                             'transformation_matrix': torch.from_numpy(scenario_batch['ego']['transformation_matrix'][tick_idx]).unsqueeze(1).to(device),
        #                             'pairwise_t_matrix': torch.from_numpy(scenario_batch['ego']['pairwise_t_matrix'][tick_idx]).unsqueeze(1).to(device),
        #                             # 'record_len': scenario_batch['ego']['record_len'][tick_idx].unsqueeze(1).to(device),
        #                             'scenario_id': scenario_batch['ego']['scenario_id'][tick_idx],
        #                             'agent_true_loc' : (scenario_batch['ego']['agent_true_loc'][tick_idx]).unsqueeze(1).to(device),
        #                             'cav_list' : (scenario_batch['ego']['cav_list'][tick_idx]),
        #                             # 'dist_to_ego' : distance_all_batch,
        #                             'single_bev' : torch.from_numpy(scenario_batch['ego']['single_bev'][tick_idx]).unsqueeze(1).to(device),
        #                             'timestamp_key' : torch.from_numpy(scenario_batch['ego']['timestamp_key'][tick_idx]).unsqueeze(1).to(device)
        #                     }
                            
        #                     batch_dict = {
        #                                     'ego': {
        #                                         k: v.unsqueeze(0) if torch.is_tensor(v) and v.ndim >= 1 else v
        #                                         for k, v in gt_tick.items()
        #                                     }
        #                                 }       ## torch.Size([1, 1, 1, 512, 512])
                            
        #                     ## model output static seg : torch.Size([1, 1, 3, 512, 512])
        #                     ## model output dynamic seg : torch.Size([1, 1, 2, 512, 512])
                            
        #                     final_loss = criterion(model_output_dict, gt_tick)
        #                     valid_ave_loss.append(final_loss.item())
                            
        #                     model_output_dict = \
        #                         opencood_val_dataset.post_process(gt_tick,
        #                                                             model_output_dict)      ## model output dict만 처리
                                
                            
        #                     ## GT input shape : torch.Size([1, 5, 1, 4, 512, 512, 3])       ## agent_nums = 5 ==> max agents pdding이 됨
        #                     ## GT static map : gt_static shape : torch.Size([1, 1, 1, 512, 512])
        #                     ## model_output_dict input shape (static_map) : torch.Size([1, 512, 512])
                    
        #                     train_utils.save_bev_seg_binary(model_output_dict,              ## logs/ys_model_xxxxxx/train_vis
        #                                                     batch_dict,
        #                                                     saved_path,
        #                                                     i,
        #                                                     tick_idx,
        #                                                     label_size=gt_tick['gt_dynamic'].shape[-1],
        #                                                     epoch = epoch)
        #                     iou_cal = cal_iou_training(batch_dict,
        #                                                     model_output_dict)

        #                     static_ave_iou.append(iou_cal[0])
        #                     dynamic_ave_iou.append(iou_cal[1])
        #                     # lane_ave_iou.append(iou_static[2])
                            
        #                 prev_encoding_result = current_encoding_result.detach()
        #                 torch.cuda.empty_cache()  # ✅ tick마다 GPU 캐시 정리

        #     valid_ave_loss = statistics.mean(valid_ave_loss)
        #     static_ave_iou = statistics.mean(static_ave_iou)
        #     # lane_ave_iou = statistics.mean(lane_ave_iou)
        #     dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

        #     print('-'*100)
        #     print('At epoch %d, the validation final_loss is %f,'
        #         'the dynamic iou is %f, t'
        #         % (epoch,
        #             valid_ave_loss,
        #             dynamic_ave_iou,
        #             ))
        #     print('-'*100)

        #     writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
        #     writer.add_scalar('Dynamic_Iou', dynamic_ave_iou, epoch)
        #     writer.add_scalar('Road_IoU', static_ave_iou, epoch)
        #         # writer.add_scalar('Lane_IoU', static_ave_iou, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        # opencood_train_dataset.reinitialize()         ## cav 순서가 바뀌어 ego가 바뀌는 상황이 없으므로 reinitialize해줄 필요없음

if __name__ == '__main__':
    main()