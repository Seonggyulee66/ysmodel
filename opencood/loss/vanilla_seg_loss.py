import torch
import torch.nn as nn
import cv2
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(logp, target, reduction='none', weight=self.ce.weight)
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
class DiceLoss(nn.Module):
    def __init__(self, class_idx=1, smooth=1.0):
        super().__init__()
        self.class_idx = class_idx  # vehicle class
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) - raw logits
        targets: (B, H, W) - long
        """
        # softmax 후 해당 클래스만 선택
        inputs = F.softmax(inputs, dim=1)[:, self.class_idx, :, :]  # (B, H, W)
        targets = (targets == self.class_idx).float()  # binary mask (B, H, W)

        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def dilate_label_tensor(label_tensor, kernel_size=3, iterations=1):
    """
    label_tensor: torch.Tensor of shape (B, H, W), dtype long
    dilation은 class==1 (vehicle class)에만 적용됨
    """
    label_np = label_tensor.cpu().numpy()  # numpy로 변환
    dilated_np = np.zeros_like(label_np)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for b in range(label_np.shape[0]):
        # class == 1 인 영역만 dilation
        vehicle_mask = (label_np[b] == 1).astype(np.uint8)
        dilated = cv2.dilate(vehicle_mask, kernel, iterations=iterations)

        # dilated == 1인 영역을 class 1로, 나머지는 원래대로
        dilated_label = label_np[b]
        dilated_label[dilated == 1] = 1
        dilated_np[b] = dilated_label

    # 다시 torch.Tensor로 변환
    return torch.from_numpy(dilated_np).to(label_tensor.device)

# def visualize_dilation(original: torch.Tensor, dilated: torch.Tensor, idx: int = 0):
#     """
#     original, dilated: (B, H, W) torch.Tensor
#     idx: 시각화할 batch index
#     """
#     orig = original[idx].cpu().numpy()
#     dila = dilated[idx].cpu().numpy()

#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(orig, cmap='gray')
#     axs[0].set_title('Original Label')
#     axs[1].imshow(dila, cmap='gray')
#     axs[1].set_title('Dilated Label')
#     for ax in axs:
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(f"dialated_fig/4_dilated_label_debug.png")
#     plt.close()
    
class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super(VanillaSegLoss, self).__init__()

        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        self.d_coe = args['d_coe']
        self.s_coe = args['s_coe']
        self.p_coe = args['p_coe']
        self.o_coe = args['o_coe']

        self.target = args['target']

        self.loss_func_static = \
            FocalLoss(gamma=2.0,
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())       ## [Class 0 weight : 1, Class 1 : self.s_weights, Class 2 : self.l_weights]
        self.loss_func_dynamic = \
            FocalLoss(gamma=2.0,weight=torch.Tensor([1.,self.d_weights]).cuda())
        
        self.dice_loss_dynamic = DiceLoss(class_idx=1)
        
        self.loss_dict = {}

    def forward(self, output_dict, gt_dict,  current_epoch=0):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """

        static_pred = output_dict['static_seg']
        dynamic_pred = output_dict['dynamic_seg']

        static_loss = torch.tensor(0, device=static_pred.device)
        dynamic_loss = torch.tensor(0, device=dynamic_pred.device)

        # during training, we only need to compute the ego vehicle's gt loss
        static_gt = gt_dict['gt_static']
        dynamic_gt = gt_dict['gt_dynamic']
        static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
        dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')
        
        if current_epoch < 50:
            dynamic_gt = dilate_label_tensor(dynamic_gt,kernel_size=5,iterations=4)
        # visualize_dilation(dynamic_gt, dilated_dynamic_gt, idx=0)
        dynamic_gt = dynamic_gt.long()
        static_gt = static_gt.long()
        
        if self.target == 'dynamic':
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            print("Foreground channel mean:", dynamic_pred[:, 1].mean().item())

            pred_dynamic_class = dynamic_pred.argmax(dim=1)
            unique_classes = torch.unique(pred_dynamic_class)
            # print(f"Dynamic prediction unique class : {unique_classes.tolist()}")
            # print(f"Dynamic GT unique classes: {torch.unique(dynamic_gt).tolist()}")
            vehicle_ratio = (dynamic_gt == 1).float().mean()
            print(f"GT Foreground(vehicle) pixel 비율: {vehicle_ratio.item():.5f}")

            dynamic_focal = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            dynamic_dice = self.dice_loss_dynamic(dynamic_pred, dynamic_gt)
            dynamic_loss = 0.5 * dynamic_focal + 0.5 * dynamic_dice

        elif self.target == 'static':
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        else:
            # dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            # dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            # static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            # static_loss = self.loss_func_static(static_pred, static_gt)
            pos_loss = output_dict['pos_loss'][0]
            assert torch.isfinite(pos_loss).all(), f"pos_loss: {pos_loss}"

            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')

            
            # print("Before loss_func_dynamic")
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            assert torch.isfinite(dynamic_loss).all(), f"dynamic_loss: {dynamic_loss}, dynamic_pred {dynamic_pred}, dynamic_gt {dynamic_gt}"

            # print("Before loss_func_static")
            static_loss = self.loss_func_static(static_pred, static_gt)
            assert torch.isfinite(static_loss).all(), f"static_loss: {static_loss}"


        offset_loss = 0
        pos_loss = output_dict['pos_loss'][0]
        offset_loss = sum(offset.abs().mean() for offset in output_dict['offsets'][0])

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss + self.p_coe * pos_loss + self.o_coe* offset_loss
        self.loss_dict.update({'total_loss': total_loss,
                               'static_loss': static_loss,
                               'dynamic_loss': dynamic_loss,
                               'pos_loss': pos_loss,
                               'off_loss' : offset_loss})

        return total_loss

    def logging(self, epoch, batch_id, scenario_id_check, batch_len, writer, pbar=None, rank=0):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']
        pos_loss = self.loss_dict['pos_loss']
        offset_loss = self.loss_dict['off_loss']
        if rank == 0:
            if pbar is None:
                print("[epoch %d][%d/%d], [Scenario Id %d] || Loss: %.4f || static Loss: %.4f"
                    " || Dynamic Loss: %.4f || Position Loss : %.4f || Offset Loss : %.4f"  % (
                        epoch, batch_id + 1, batch_len, scenario_id_check,
                        total_loss.item(), static_loss.item(), dynamic_loss.item(), pos_loss.item(), offset_loss.item()))
            else:
                pbar.set_description("[epoch %d][%d/%d], [Scenario Id %d] || Loss: %.4f || static Loss: %.4f"
                    " || Dynamic Loss: %.4f || Postion Loss : %.4f || Offset Loss : %.4f" % (
                        epoch, batch_id + 1, batch_len,scenario_id_check,
                        total_loss.item(), static_loss.item(), dynamic_loss.item(), pos_loss.item(),  offset_loss.item()))


            writer.add_scalar('Static_loss', static_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Position_loss', pos_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Offset_loss', offset_loss.item(),
                            epoch*batch_len + batch_id)




