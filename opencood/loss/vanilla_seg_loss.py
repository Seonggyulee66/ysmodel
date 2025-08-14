import torch
import torch.nn as nn
import cv2
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------- Tversky (binary) ----------
def tversky_binary_from_probs(p_fg, t_fg, alpha=0.3, beta=0.7, eps=1e-6):
    """
    p_fg, t_fg: [B,1,H,W], p_fg는 sigmoid 확률(0~1), t_fg는 {0,1}
    반환: (1 - Tversky) 의 배치 평균 = loss
    """
    p = p_fg.clamp(eps, 1 - eps)
    t = t_fg.float()
    tp = (p * t).sum(dim=(2,3))
    fp = (p * (1 - t)).sum(dim=(2,3))
    fn = ((1 - p) * t).sum(dim=(2,3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()

# ---------- Tversky (multiclass) ----------
def tversky_multiclass_from_probs(p, t, alpha=0.3, beta=0.7, eps=1e-6, ignore_index=None):
    """
    p: [B,C,H,W] softmax 확률, t: [B,H,W] long
    반환: (1 - per-class tversky).mean() = loss
    """
    B, C, H, W = p.shape
    p = p.clamp(eps, 1 - eps)
    t_oh = F.one_hot(t.clamp_min(0), num_classes=C).permute(0,3,1,2).float()

    if ignore_index is not None:
        mask = (t != ignore_index).float().unsqueeze(1)  # [B,1,H,W]
        p = p * mask
        t_oh = t_oh * mask

    tp = (p * t_oh).sum(dim=(0,2,3))      # per-class
    fp = (p * (1 - t_oh)).sum(dim=(0,2,3))
    fn = ((1 - p) * t_oh).sum(dim=(0,2,3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)  # (C,)
    return 1.0 - tversky.mean()

# ---------- Focal (binary) ----------
def focal_binary_with_logits(logits, targets, gamma=2.0, alpha=0.25, weight=None, eps=1e-8):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probas = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probas, 1 - probas)
    focal = (alpha * (1 - pt) ** gamma) * bce

    if weight is not None:
        focal = focal * weight  # 여기서 weight_map 곱해줌

    return focal.mean()
# ---------- Focal (multiclass) ----------
def focal_multiclass_with_logits(logits, target, gamma=1.0, class_weight=None, ignore_index=None):
    """
    logits: [B,C,H,W], target: [B,H,W] (long)
    CE 기반 FocalLoss: FL = (1 - p_t)^gamma * CE
    """
    log_p = F.log_softmax(logits, dim=1)           # [B,C,H,W]
    p = log_p.exp()

    if ignore_index is not None:
        valid = (target != ignore_index)
        target = target.clone()
        target[~valid] = 0  # 임시 값
    else:
        valid = torch.ones_like(target, dtype=torch.bool)

    pt = p.gather(1, target.unsqueeze(1)).squeeze(1)       # [B,H,W]
    ce = F.nll_loss(
        log_p, target,
        weight=class_weight,
        ignore_index=ignore_index if ignore_index is not None else -1000,  # 무시 안 할 때 영향 없도록
        reduction="none"
    )  # [B,H,W]

    fl = (1 - pt).pow(gamma) * ce
    fl = fl[valid].mean() if valid.any() else ce.mean()
    return fl

# ---------- VanillaSegLoss (Tversky + Focal만) ----------
class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        self.d_coe = args['d_coe']
        self.s_coe = args['s_coe']
        self.p_coe = args['p_coe']
        self.o_coe = args['o_coe']

        self.target = args['target']

        # 조합 비율
        self.lambda_tversky = args.get('lambda_tversky', 0.6)
        self.lambda_focal   = args.get('lambda_focal',   0.4)

        # Tversky (리콜↑이면 beta↑)
        self.tversky_alpha = args.get('tversky_alpha', 0.3)
        self.tversky_beta  = args.get('tversky_beta',  0.75)   # <- 0.7 -> 0.75로 미세 상향 권장
        self.tversky_eps   = args.get('tversky_eps',   1e-6)   # 안정화용

        # Focal (under-confidence 완화)
        self.focal_gamma_bin = args.get('focal_gamma_bin', 1.5)   # <- 2.0 에서 1.5로 완화
        self.focal_alpha_bin = args.get('focal_alpha_bin', None)  # <- 동적 alpha를 쓸 것이므로 기본 None
        self.focal_gamma_mc  = args.get('focal_gamma_mc',  1.0)

        # Effective number weight (binary에 적용)
        self.use_effective_num = args.get('use_effective_num', True)
        self.en_beta = args.get('effective_num_beta', 0.9999)

        # static class weight / ignore
        self.ignore_index = args.get('ignore_index', None)
        s_w = float(args.get('s_weights', 30.0))
        l_w = float(args.get('l_weights', 50.0))
        self.register_buffer('static_class_weight',
            torch.tensor([1.0, s_w, l_w], dtype=torch.float32))

        self.loss_dict = {}

    @staticmethod
    def _effective_num_weights(pos_count, neg_count, beta=0.9999, eps=1e-8, device='cpu'):
        # pos/neg 개수에서 effective number 기반 w_pos, w_neg 산출
        # w = (1 - beta) / (1 - beta^n)
        n_pos = torch.as_tensor(pos_count, dtype=torch.float32, device=device).clamp_min(1.0)
        n_neg = torch.as_tensor(neg_count, dtype=torch.float32, device=device).clamp_min(1.0)
        en_pos = (1.0 - beta**n_pos).clamp_min(eps)
        en_neg = (1.0 - beta**n_neg).clamp_min(eps)
        w_pos = (1.0 - beta) / en_pos
        w_neg = (1.0 - beta) / en_neg
        # 정규화(평균=1 근처)
        m = 0.5 * (w_pos + w_neg)
        return (w_pos / m).detach(), (w_neg / m).detach()

    def forward(self, output_dict, gt_dict):
        device = self.static_class_weight.device

        static_loss  = torch.zeros((), device=device)
        dynamic_loss = torch.zeros((), device=device)

        # ---------- Dynamic (binary) ----------
        if self.target in ('dynamic', 'both'):
            dynamic_pred = output_dict['dynamic_seg']        # [B,L,2,H,W]
            dynamic_gt   = gt_dict['gt_dynamic']             # [B,L,H,W]

            dyn_logits = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dyn_target = rearrange(dynamic_gt,  'b l h w -> (b l) h w').long()

            # FG logit/prob
            dyn_logit_fg = dyn_logits[:, 1:2]                     # [BL,1,H,W]
            dyn_prob_fg  = torch.sigmoid(dyn_logit_fg)            # [BL,1,H,W]
            dyn_target_fg = (dyn_target == 1).unsqueeze(1).float()

            # ---- 동적 alpha (배치 FG 비율 기반) ----
            # p_fg = (#FG pixel) / (#valid pixel)
            valid = torch.ones_like(dyn_target_fg, dtype=torch.bool, device=dyn_target_fg.device)
            fg_count = dyn_target_fg.sum()
            total_count = valid.sum().clamp_min(1)
            p_fg = (fg_count.float() / total_count.float()).clamp(0.0001, 0.9999)
            # focal alpha는 양성 가중; 희소 FG면 alpha를 크게 -> alpha = 1 - p_fg가 자연스러움
            dyn_alpha = 1.0 - p_fg.item() if self.focal_alpha_bin is None else float(self.focal_alpha_bin)

            # ---- effective number class weight (선택) ----
            if self.use_effective_num:
                w_pos, w_neg = self._effective_num_weights(
                    pos_count=fg_count.item(),
                    neg_count=(total_count - fg_count).item(),
                    beta=self.en_beta,
                    device=dyn_target_fg.device
                )
                # per-pixel weight map: FG에 w_pos, BG에 w_neg 적용
                weight_map = torch.where(dyn_target_fg.bool(), w_pos, w_neg)
            else:
                weight_map = None

            # ---- Tversky + Focal (binary) ----
            # Tversky (from probs)
            dyn_tv = tversky_binary_from_probs(
                dyn_prob_fg.clamp(self.tversky_eps, 1 - self.tversky_eps),
                dyn_target_fg,
                alpha=self.tversky_alpha,
                beta=self.tversky_beta
            )
            # Focal (with logits) + 동적 alpha + (선택)weight_map
            dyn_fl = focal_binary_with_logits(
                dyn_logit_fg,
                dyn_target_fg,
                gamma=self.focal_gamma_bin,
                alpha=dyn_alpha,
                weight=weight_map    # <- 구현체가 weight 인자를 지원하면 연결, 아니면 내부 곱 처리
            )
            dynamic_loss = self.lambda_tversky * dyn_tv + self.lambda_focal * dyn_fl

        # ---------- Static (multiclass) ----------
        if self.target in ('static', 'both'):
            static_pred = output_dict['static_seg']          # [B,L,3,H,W]
            static_gt   = gt_dict['gt_static']               # [B,L,H,W]

            sta_logits = rearrange(static_pred, 'b l c h w -> (b l) c h w')  # [BL,3,H,W]
            sta_target = rearrange(static_gt,  'b l h w -> (b l) h w').long()

            sta_prob = F.softmax(sta_logits, dim=1).clamp(self.tversky_eps, 1 - self.tversky_eps)
            sta_tv = tversky_multiclass_from_probs(
                sta_prob, sta_target,
                alpha=self.tversky_alpha, beta=self.tversky_beta,
                ignore_index=self.ignore_index, eps=self.tversky_eps
            )
            sta_fl = focal_multiclass_with_logits(
                sta_logits, sta_target,
                gamma=self.focal_gamma_mc,
                class_weight=self.static_class_weight.to(sta_logits.device),
                ignore_index=self.ignore_index
            )
            static_loss = self.lambda_tversky * sta_tv + self.lambda_focal * sta_fl

        # ---------- Aux losses (안전 대체) ----------
        pos_loss = output_dict.get('pos_loss', [torch.zeros((), device=device)])[0]
        if isinstance(pos_loss, (list, tuple)):
            pos_loss = pos_loss[0]
        offsets = output_dict.get('offsets', [[torch.zeros((), device=device)]])[0]
        off_loss = torch.stack([o.abs().mean() if o.numel() > 0 else torch.zeros((), device=device)
                                for o in offsets]).mean() if isinstance(offsets, (list, tuple)) else torch.zeros((), device=device)

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss + self.p_coe * pos_loss + self.o_coe * off_loss

        # 로깅용 (detach)
        self.loss_dict.update({
            'total_loss': total_loss.detach(),
            'static_loss': static_loss.detach(),
            'dynamic_loss': dynamic_loss.detach(),
            'pos_loss': pos_loss.detach() if torch.is_tensor(pos_loss) else torch.tensor(float(pos_loss), device=device),
            'off_loss': off_loss.detach()
        })
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




