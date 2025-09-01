import glob
import importlib
import sys
import yaml
import os
import re
import random
from datetime import datetime

import cv2
import torch
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR,
    OneCycleLR, CosineAnnealingWarmRestarts, LambdaLR
)
from timm.scheduler.cosine_lr import CosineLRScheduler

from opencood.tools.multi_gpu_utils import get_dist_info
from opencood.utils.common_utils import torch_tensor_to_numpy

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(
            os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch),
            map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

        del checkpoint

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py
    """
    # 안전장치: 멀티프로세스 __pycache__ 레이스 회피
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    importlib.invalidate_caches()

    backbone_name   = hypes['model']['core_method']
    backbone_config = hypes['model']['args']
    model_filename  = "opencood.models." + backbone_name

    # rank 파악
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", "0"))
    except Exception:
        rank = int(os.environ.get("RANK", "0"))

    def _safe_import(name):
        try:
            return importlib.import_module(name)
        except Exception as e:
            spec   = importlib.util.find_spec(name)
            origin = getattr(spec, "origin", None)
            print(f"[IMPORT-DEBUG][rank={rank}] name={name} origin={origin} err={repr(e)}", file=sys.stderr, flush=True)
            # 오염 quick check (rank0만)
            if origin and rank == 0:
                try:
                    with open(origin, "rb") as f:
                        data = f.read(256*1024)  # 처음 256KB만 체크해도 충분
                    if b"\x00" in data:
                        print(f"[IMPORT-DEBUG] NUL(\\x00) bytes detected in {origin}", file=sys.stderr, flush=True)
                except Exception as ee:
                    print(f"[IMPORT-DEBUG] failed to read {origin}: {ee}", file=sys.stderr, flush=True)
            raise

    # rank0 선임포트 → barrier → 나머지 임포트 (레이스 방지)
    if dist.is_available() and dist.is_initialized():
        if rank == 0:
            model_lib = _safe_import(model_filename)
        if dist.get_world_size() > 1:
            dist.barrier()
        if rank != 0:
            model_lib = _safe_import(model_filename)
    else:
        model_lib = _safe_import(model_filename)

    # 클래스 찾기
    model = None
    target_model_name = backbone_name.replace('_', '')
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
            break

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class called %s (case-insensitive)'
              % (model_filename, target_model_name))
        sys.exit(1)

    instance = model(backbone_config)
    return instance



def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


# def setup_optimizer(hypes, model):
#     """
#     Create optimizer corresponding to the yaml file

#     Parameters
#     ----------
#     hypes : dict
#         The training configurations.
#     model : opencood model
#         The pytorch model
#     """
#     method_dict = hypes['optimizer']
#     optimizer_method = getattr(optim, method_dict['core_method'], None)
#     print('optimizer method is: %s' % optimizer_method)

#     if not optimizer_method:
#         raise ValueError('{} is not supported'.format(method_dict['name']))
#     if 'args' in method_dict:
#         return optimizer_method(filter(lambda p: p.requires_grad,
#                                        model.parameters()),
#                                 lr=method_dict['lr'],
#                                 **method_dict['args'])
#     else:
#         return optimizer_method(filter(lambda p: p.requires_grad,
#                                        model.parameters()),
#                                 lr=method_dict['lr'])

def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('optimizer method is: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['core_method']))

    # ✅ weight decay 분리 기준 정의
    no_decay_keywords = ['bias', 'norm', 'to_q', 'to_k', 'to_v']
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name.lower() for key in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # ✅ weight decay 값 가져오기
    weight_decay = method_dict['args'].get('weight_decay', 0.0)
    lr = method_dict['lr']
    args = method_dict['args']
    args = {k: v for k, v in args.items() if k != 'weight_decay'}  # 제거

    # ✅ optimizer 생성
    return optimizer_method(
        [
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': weight_decay}
        ],
        lr=lr,
        **args
    )


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    n_iter_per_epoch : int
        Iterations per epoech.
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        print('StepLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        print('MultiStepLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch 
        warmup_lr = lr_schedule_config['warmup_lr'] 
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    else:
        sys.exit('Unidentified scheduler')

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)

### original
# def save_bev_seg_binary(output_dict,
#                         batch_dict,
#                         output_dir,
#                         batch_iter,
#                         epoch,
#                         test=False):
#     """
#     Save the bev segmetation results during training.

#     Parameters
#     ----------
#     batch_dict: dict
#         The data that contains the gt.

#     output_dict : dict
#         The output directory with predictions.

#     output_dir : str
#         The output directory.

#     batch_iter : int
#         The batch index.

#     epoch : int
#         The training epoch.

#     test: bool
#         Whether this is during test or train.
#     """

#     if test:
#         output_folder = os.path.join(output_dir, 'test_vis')
#     else:
#         output_folder = os.path.join(output_dir, 'train_vis', str(epoch))

#     if not os.path.exists(output_folder):
#         try:
#             os.makedirs(output_folder)
#         except FileExistsError:
#             pass

#     batch_size = batch_dict['ego']['gt_static'].shape[0]

#     for i in range(batch_size):
#         # gt_static_origin = \
#         #     batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[i, 0]
#         # gt_static = np.zeros((gt_static_origin.shape[0],
#         #                       gt_static_origin.shape[1],
#         #                       3), dtype=np.uint8)
#         # gt_static[gt_static_origin == 1] = np.array([88, 128, 255])
#         # gt_static[gt_static_origin == 2] = np.array([244, 148, 0])

#         gt_dynamic = \
#             batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
#         gt_dynamic = np.array(gt_dynamic * 255., dtype=np.uint8)

#         # pred_static_origin = \
#         #     output_dict['static_map'].detach().cpu().data.numpy()[i]
#         # pred_static = np.zeros((pred_static_origin.shape[0],
#         #                         pred_static_origin.shape[1],
#         #                         3), dtype=np.uint8)
#         # pred_static[pred_static_origin == 1] = np.array([88, 128, 255])
#         # pred_static[pred_static_origin == 2] = np.array([244, 148, 0])

#         pred_dynamic = \
#             output_dict['dynamic_map'].detach().cpu().data.numpy()[i]
#         pred_dynamic = np.array(pred_dynamic * 255., dtype=np.uint8)

#         # try to find the right index for raw image visualization
#         index = i
#         if 'record_len' in batch_dict['ego']:
#             cum_sum_len = \
#                 [0] + list(np.cumsum(
#                     torch_tensor_to_numpy(batch_dict['ego']['record_len'])))
#             index = cum_sum_len[i]
#         # (M, H, W, 3)
#         raw_images = \
#             batch_dict['ego']['inputs'].detach().cpu().data.numpy()[index, 0]       ## shape (4, label_size, label_size, 3)
#         visualize_summary = np.zeros((raw_images[0].shape[0] * 2,
#                                       raw_images[0].shape[1] * 4,
#                                       3),
#                                      dtype=np.uint8)
#         for j in range(raw_images.shape[0]):
#             raw_image = 255 * ((raw_images[j] * STD) + MEAN)
#             raw_image = np.array(raw_image, dtype=np.uint8)
#             # rgb = bgr
#             raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
#             visualize_summary[:raw_image.shape[0],
#             j * raw_image.shape[1]:
#             (j + 1) * raw_image.shape[1],
#             :] = raw_image
#         # draw gts on the visualization summary
#         gt_dynamic = cv2.resize(gt_dynamic, (raw_image.shape[0],
#                                              raw_image.shape[1]))
#         pred_dynamic = cv2.resize(pred_dynamic, (raw_image.shape[0],
#                                                  raw_image.shape[1]))
#         # gt_static = cv2.resize(gt_static, (raw_image.shape[0],
#         #                                    raw_image.shape[1]))
#         # pred_static = cv2.resize(pred_static, (raw_image.shape[0],
#         #                                        raw_image.shape[1]))

#         visualize_summary[raw_image.shape[0]:, :raw_image.shape[1], :] = \
#             cv2.cvtColor(gt_dynamic, cv2.COLOR_GRAY2BGR)
#         visualize_summary[raw_image.shape[0]:,
#         raw_image.shape[1]:2 * raw_image.shape[1], :] = \
#             cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)
#         # visualize_summary[raw_image.shape[0]:,
#         # 2 * raw_image.shape[1]:3 * raw_image.shape[1], :] = gt_static
#         # visualize_summary[raw_image.shape[0]:,
#         # 3 * raw_image.shape[1]:4 * raw_image.shape[1], :] = pred_static

#         cv2.imwrite(os.path.join(output_folder, '%d_%d_vis.png')
#                     % (batch_iter, i), visualize_summary)


import os
import numpy as np
import cv2

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def torch_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def save_bev_seg_binary(output_dict,
                        batch_dict,
                        output_dir,
                        chunk_id,
                        batch_iter,
                        epoch,
                        label_size,
                        test=False):
    """
    Save the BEV segmentation results during training, including all agents' camera inputs.
    output_dict -> post_process된 model output [torch.Size([1, label_size, label_size])]
    batch_dict -> batch_dict['ego'][~~] 로 정리된 dictionary [ex) ['inputs'] -> torch.Size([1, 5, 1, 4, label_size, 512, 3])]
    """

    if test:
        output_folder = os.path.join(output_dir, 'test_vis')
    else:
        output_folder = os.path.join(output_dir, 'train_vis', str(epoch))

    os.makedirs(output_folder, exist_ok=True)

    batch_size = batch_dict['ego']['gt_static'].shape[2] 
    num_agents = 0
    
    for i in range(batch_dict['ego']['inputs'].shape[1]):
        if torch.all(batch_dict['ego']['inputs'][0,i,0,0,:] == 0.):
            continue
        num_agents += 1
    
    num_cameras = batch_dict['ego']['inputs'].shape[3]

    for i in range(batch_size):
        gt_dynamic = batch_dict['ego']['gt_dynamic'].detach().cpu().numpy()[0, i, 0]
        gt_dynamic = np.array(gt_dynamic * 255., dtype=np.uint8)

        pred_dynamic = output_dict['dynamic_map'].detach().cpu().numpy()[i]
        pred_dynamic = np.array(pred_dynamic * 255., dtype=np.uint8)

        # Find correct index if 'record_len' exists
        index = i
        if 'record_len' in batch_dict['ego']:
            cum_sum_len = [0] + list(np.cumsum(
                torch_tensor_to_numpy(batch_dict['ego']['record_len'])))
            index = cum_sum_len[i]

        canvas_w = label_size * num_cameras  # 각 카메라당 label_size 픽셀 너비
        bev_height = 256              # BEV row 높이
        camera_height = label_size * num_agents
        dynmap_height = label_size
        canvas_h_total = camera_height + bev_height + dynmap_height
        visualize_summary = np.zeros((canvas_h_total, canvas_w, 3), dtype=np.uint8)

        # 1~3행: Camera views (Agent 순서대로 위에서부터 나열)
        for agent_id in range(num_agents):
            raw_images = batch_dict['ego']['inputs'][0, agent_id, i]  # shape: (4, label_size, label_size, 3)
            for cam_id in range(num_cameras):
                raw_image = raw_images[cam_id].detach().cpu().numpy()
                raw_image = 255 * ((raw_image * STD) + MEAN)
                raw_image = np.array(raw_image, dtype=np.uint8)
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

                y_start = agent_id * label_size
                x_start = cam_id * label_size
                visualize_summary[y_start:y_start + label_size, x_start:x_start + label_size, :] = raw_image

        # 4행: BEV maps (각 agent의 BEV → 하나의 행에 가로 분할 배치)
        bev_row = np.zeros((bev_height, canvas_w, 3), dtype=np.uint8)
        x_offset = 0
        for agent_id in range(num_agents):
            target_width = canvas_w // num_agents
            if agent_id == num_agents - 1:
                target_width = canvas_w - x_offset  # 마지막 agent는 남은 폭 전부

            bev_map = batch_dict['ego']['single_bev'][0, agent_id, 0].detach().cpu().numpy()
            bev_map = np.array(bev_map * 255., dtype=np.uint8)
            bev_map = cv2.resize(bev_map, (target_width, bev_height), interpolation=cv2.INTER_NEAREST)
            bev_map = cv2.cvtColor(bev_map, cv2.COLOR_GRAY2BGR)

            bev_row[:, x_offset:x_offset + target_width, :] = bev_map
            x_offset += target_width

        visualize_summary[camera_height:camera_height + bev_height, :canvas_w, :] = bev_row

        # 5행: dynamic map (GT vs Prediction)
        gt_dynamic = cv2.resize(gt_dynamic, (canvas_w // 2, dynmap_height))
        pred_dynamic = cv2.resize(pred_dynamic, (canvas_w // 2, dynmap_height))
        gt_dynamic = cv2.cvtColor(gt_dynamic, cv2.COLOR_GRAY2BGR)
        pred_dynamic = cv2.cvtColor(pred_dynamic, cv2.COLOR_GRAY2BGR)

        y_dyn_start = camera_height + bev_height
        visualize_summary[y_dyn_start:y_dyn_start + dynmap_height, :canvas_w // 2, :] = gt_dynamic
        visualize_summary[y_dyn_start:y_dyn_start + dynmap_height, canvas_w // 2:canvas_w, :] = pred_dynamic

        # Save
        save_path = os.path.join(output_folder, f'{int(chunk_id)}_{batch_iter}_{i}_vis.png')
        cv2.imwrite(save_path, visualize_summary)



def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False