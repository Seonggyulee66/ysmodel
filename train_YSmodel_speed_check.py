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
import torch, torch.nn.functional as F
import torch
import torch.nn.functional as F
import time

# === 프로파일 유틸 ===
import time
import statistics

class GPUTimer:
    """CUDA 구간 측정용. with 블록으로 사용."""
    def __init__(self, enable=True):
        self.enable = enable and torch.cuda.is_available()
        if self.enable:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        self.ms = 0.0

    def __enter__(self):
        if self.enable:
            torch.cuda.synchronize()
            self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enable:
            self.end.record()
            torch.cuda.synchronize()
            self.ms = self.start.elapsed_time(self.end)  # milliseconds

class CPUTimer:
    """CPU 구간 측정용. with 블록으로 사용."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.ms = (time.perf_counter() - self.t0) * 1000.0

class TickProfiler:
    """tick 단위 측정치 누적 + 통계."""
    KEYS = ["data", "h2d", "encode", "forward", "loss", "backward", "optim", "step_total"]

    def __init__(self):
        self.hist = {k: [] for k in self.KEYS}
        self._last_data_t0 = None  # data time 측정을 위한 구간 시작 지점

    def mark_data_start(self):
        # 다음 tick의 "data" 구간 시작
        self._last_data_t0 = time.perf_counter()

    def add_ms(self, key, ms):
        self.hist[key].append(float(ms))

    def summary(self):
        def stats(x):
            return dict(
                mean=statistics.mean(x) if x else 0.0,
                p50=statistics.median(x) if x else 0.0,
                max=max(x) if x else 0.0,
                n=len(x),
            )
        return {k: stats(v) for k, v in self.hist.items()}

    def pretty(self):
        s = []
        summ = self.summary()
        for k in self.KEYS:
            st = summ[k]
            s.append(f"{k:>9s}: mean {st['mean']:7.2f} ms | p50 {st['p50']:7.2f} | max {st['max']:7.2f} | n={st['n']}")
        return "\n".join(s)


def unwrap_module(m):
    # DDP/DP 래핑이면 .module, 아니면 자기 자신
    return m.module if hasattr(m, "module") else m

@torch.no_grad()
def quick_monitor_v2(logits, target, th=0.3):
    """
    logits: (B,C,H,W) or (B,L,C,H,W)  where C==2 (bg,fg) or C==1 (fg-logit)
    target: (B,H,W) or (B,1,H,W) or (B,L,H,W)
    """
    # 1) (B,L,...) 이면 평탄화
    if logits.dim() == 5:  # (B,L,C,H,W)
        B, L, C, H, W = logits.shape
        logits = logits.reshape(B * L, C, H, W)
        if target.dim() == 4 and target.size(1) == L:  # (B,L,H,W)
            target = target.reshape(B * L, H, W)
        elif target.dim() == 5 and target.size(2) == 1:  # (B,L,1,H,W)
            target = target.reshape(B * L, 1, H, W)

    # 2) 확률 계산
    if logits.size(1) == 1:
        prob_fg = torch.sigmoid(logits[:, 0])            # (B*,H,W)
    else:
        prob = torch.softmax(logits, dim=1)              # (B*,2,H,W)
        prob_fg = prob[:, 1]                             # fg 확률

    # 3) 타깃 정리
    if target.dim() == 4 and target.size(1) == 1:  # (B*,1,H,W)
        target = target[:, 0]
    elif target.dim() == 3:
        pass
    else:
        target = target.squeeze()

    y = (target > 0).bool()
    pred = (prob_fg >= th)

    tp = (pred & y).sum().item()
    fp = (pred & ~y).sum().item()
    fn = (~pred & y).sum().item()

    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    PPR = pred.float().mean().item()      # 임계값 적용 후 양성 비율
    p_mean = prob_fg.mean().item()        # 임계값 없이 평균 양성확률
    y_prev = y.float().mean().item()      # 라벨의 실제 양성 비율

    print(f"[th={th:.2f}] P={P:.3f} R={R:.3f} PPR={PPR:.3f} p_mean={p_mean:.3f} y_prev={y_prev:.3f}")
    return dict(P=P, R=R, PPR=PPR, p_mean=p_mean, y_prev=y_prev, tp=tp, fp=fp, fn=fn)


@torch.no_grad()
def sweep_best_threshold(logits, target, ths=None):
    if ths is None:
        ths = torch.linspace(0.05, 0.95, 19, device=logits.device)
    best, bestF1 = None, -1
    for th in ths:
        m = quick_monitor_v2(logits, target, float(th))
        F1 = 2*m["P"]*m["R"] / max(m["P"]+m["R"], 1e-6)
        if F1 > bestF1:
            bestF1, best = F1, {"th": float(th), "F1": F1, **m}
    print(f"best th={best['th']:.2f} F1={best['F1']:.3f} P={best['P']:.3f} R={best['R']:.3f}")
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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        torch.nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[opt.gpu],
                                                    find_unused_parameters=False,   # ← 여기만 False로
                                                    broadcast_buffers=False)        # ← BN을 eval로 고정했으니 버퍼 브로드캐스트도 꺼서 통신 감소(선택)
                                                    # static_graph=True,)            # ← 그래프/사용 파라미터 구성이 '항상' 동일하면 켜면 추가로 빨라짐(선택))
    
    
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)

   # (중략) main() 내부

    writer = SummaryWriter(saved_path)
    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # lr scheduler setup
    epoches = hypes['train_params']['epoches']
    num_steps = len(train_loader)
    n_iter_per_epoch = num_steps * hypes['chunk_size']  # 예: 162 * 5 = 810
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch)
    accum_steps = 3                    # 예: tick 6개마다 1 step
    global_micro = 0
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    
    # ★ 추가: 프로파일러
    epoch_prof = None  # 에폭마다 재생성
    LOG_EVERY = 50     # tick 단위 로그 주기 (원하면 조절)

    # (중략) 스케줄러 준비 직후
    scheduler.step_update(0)

    # (중략) 임계값 초기화 등 이후

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # ★ 추가: 에폭 프로파일러 새로 시작
        epoch_prof = TickProfiler()

        # (중략) 러닝레이트 출력, sampler epoch 설정 등

        if opt.rank == 0:
            pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        # ★ 추가: 첫 data 계측 시작 기준점
        epoch_prof.mark_data_start()

        for i, scenario_batch in enumerate(train_loader):
            core = unwrap_module(model)
            scenario_id_check = scenario_batch['ego']['scenario_id'][0]
            model.train()

            prev_encoding_result = None
            record_len = scenario_batch['ego']['record_len']

            # ── tick 루프
            for tick_idx in range(record_len):

                # ===== data 구간 =====
                # data 구간은 "이 tick을 시작하기까지 걸린 시간"으로 간주 (이전 tick의 optim 이후~여기까지)
                if epoch_prof._last_data_t0 is not None:
                    data_ms = (time.perf_counter() - epoch_prof._last_data_t0) * 1000.0
                    epoch_prof.add_ms("data", data_ms)

                # ===== h2d (CPU→GPU) =====
                with GPUTimer() as t_h2d:
                    tick_input_images  = scenario_batch['ego']['inputs'][tick_idx].to(device, non_blocking=True)
                    tick_input_intrins = scenario_batch['ego']['intrinsic'][tick_idx].to(device, non_blocking=True)
                    tick_input_extrins = scenario_batch['ego']['extrinsic'][tick_idx].to(device, non_blocking=True)
                    tick_input_locs    = scenario_batch['ego']['agent_true_loc'][tick_idx].to(device, non_blocking=True)

                    # Batch=1 가정 → encoding 입력 차원 맞춤
                    tick_input_images  = tick_input_images.unsqueeze(1)
                    tick_input_intrins = tick_input_intrins.unsqueeze(1)
                    tick_input_extrins = tick_input_extrins.unsqueeze(1)
                    tick_input_locs    = tick_input_locs.unsqueeze(1)
                epoch_prof.add_ms("h2d", t_h2d.ms)

                # ===== 전체 tick 시간 측정 시작 =====
                with CPUTimer() as t_step_total:

                    # ===== encode =====
                    with GPUTimer() as t_enc:
                        current_encoding_result = core.encoding(
                            tick_input_images, tick_input_intrins, tick_input_extrins, tick_input_locs, is_train=True
                        )
                        assert torch.isfinite(current_encoding_result).all(), "NaN in encoding!"
                    epoch_prof.add_ms("encode", t_enc.ms)

                    # ===== forward =====
                    with GPUTimer() as t_fwd:
                        if prev_encoding_result is None:
                            model_output_dict = model(
                                current_encoding_result, current_encoding_result, bool_prev_pos_encoded=False
                            )
                        else:
                            model_output_dict = model(
                                current_encoding_result, prev_encoding_result, bool_prev_pos_encoded=True
                            )
                    epoch_prof.add_ms("forward", t_fwd.ms)

                    # ===== loss =====
                    # GT dict 구성 (tick 단위)
                    gt_tick = {
                        'gt_static'  : torch.from_numpy(scenario_batch['ego']['gt_static'][tick_idx]).unsqueeze(1).to(device, non_blocking=True),
                        'gt_dynamic' : torch.from_numpy(scenario_batch['ego']['gt_dynamic'][tick_idx]).unsqueeze(1).to(device, non_blocking=True),
                    }

                    with GPUTimer() as t_loss:
                        final_loss = criterion(model_output_dict, gt_tick)
                    epoch_prof.add_ms("loss", t_loss.ms)

                    # ===== backward =====
                    with GPUTimer() as t_bwd:
                        if opt.half:
                            scaler.scale(final_loss).backward()
                        else:
                            final_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    epoch_prof.add_ms("backward", t_bwd.ms)

                    # ===== optim (step + zero + scaler.update) =====
                    with GPUTimer() as t_opt:
                        if opt.half:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    epoch_prof.add_ms("optim", t_opt.ms)

                    # 스케줄러/전역 카운터
                    global_micro += 1
                    global_update += 1
                    scheduler.step_update(global_update)

                    prev_encoding_result = current_encoding_result.detach()

                # ===== tick total =====
                epoch_prof.add_ms("step_total", t_step_total.ms)

                # 다음 tick을 위한 data 계측 시작 기준점 설정
                epoch_prof.mark_data_start()

                # 간이 로그 (tick마다 찍고 싶으면 LOG_EVERY=1로)
                if (global_micro % LOG_EVERY == 0) and (opt.rank == 0):
                    msg = (
                        f"[E{epoch} T{tick_idx:03d}] "
                        f"data {data_ms:6.1f} | h2d {t_h2d.ms:6.1f} | enc {t_enc.ms:6.1f} | fwd {t_fwd.ms:6.1f} | "
                        f"loss {t_loss.ms:6.1f} | bwd {t_bwd.ms:6.1f} | opt {t_opt.ms:6.1f} | step {t_step_total.ms:6.1f}  "
                        f"|| loss={float(final_loss):.4f}"
                    )
                    print(msg)

                    # TensorBoard(옵션)
                    writer.add_scalar('time/data_ms', data_ms, global_micro)
                    writer.add_scalar('time/h2d_ms', t_h2d.ms, global_micro)
                    writer.add_scalar('time/encode_ms', t_enc.ms, global_micro)
                    writer.add_scalar('time/forward_ms', t_fwd.ms, global_micro)
                    writer.add_scalar('time/loss_ms', t_loss.ms, global_micro)
                    writer.add_scalar('time/backward_ms', t_bwd.ms, global_micro)
                    writer.add_scalar('time/optim_ms', t_opt.ms, global_micro)
                    writer.add_scalar('time/step_total_ms', t_step_total.ms, global_micro)

            # 시나리오 단위 로그 (원래 있던 출력 유지)
            print(f"[{epoch} / {len(train_loader)}]  || Scenario {scenario_id_check} for {record_len} ticks | Loss : {final_loss:.4f}")

            if opt.rank == 0:
                criterion.logging(epoch, i, scenario_id_check, len(train_loader), writer, pbar=pbar2, rank=opt.rank)
                pbar2.update(1)
                for lr_idx, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'lr_{lr_idx}', param_group["lr"], global_micro)

        # ── 에폭 말: 요약 통계 출력
        if opt.rank == 0:
            print("\n=== Epoch Profiling Summary (ms) ===")
            print(epoch_prof.pretty())
            # 원한다면 요약도 TensorBoard에 기록
            summ = epoch_prof.summary()
            for k, st in summ.items():
                writer.add_scalar(f'epoch_time/{k}_mean_ms', st['mean'], epoch)
                writer.add_scalar(f'epoch_time/{k}_p50_ms',  st['p50'],  epoch)
                writer.add_scalar(f'epoch_time/{k}_max_ms',  st['max'],  epoch)

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

        # if epoch % hypes['train_params']['save_freq'] == 0:
        #     torch.save(model_without_ddp.state_dict(),
        #                os.path.join(saved_path,
        #                             'net_epoch%d.pth' % (epoch + 1)))

        # opencood_train_dataset.reinitialize()         ## cav 순서가 바뀌어 ego가 바뀌는 상황이 없으므로 reinitialize해줄 필요없음

if __name__ == '__main__':
    main()