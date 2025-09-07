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
    
# ---------- VanillaSegLoss (Tversky + Focal만) ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ----- weights / coeffs -----
        self.d_weights = float(args['d_weights'])
        self.s_weights = float(args['s_weights'])
        self.l_weights = 50 if 'l_weights' not in args else float(args['l_weights'])

        self.d_coe = float(args['d_coe'])
        self.s_coe = float(args['s_coe'])
        self.p_coe = float(args['p_coe'])
        self.o_coe = float(args['o_coe'])

        self.target = args['target']

        # ----- CE options -----
        self.ignore_index = int(args.get('ignore_index', 255))
        self.ce_ohem_pct = args.get('ce_ohem_pct', None)           # 0.2 -> 상위 20%만 평균
        self.ce_label_smoothing = float(args.get('ce_label_smoothing', 0.0))
        self.ce_temperature = float(args.get('ce_temperature', 1.5))# 로짓 온도(과신 억제)
        self.ce_cap = float(args.get('ce_cap', 20.0))              # 픽셀당 손실 상한

        # calibration (dynamic에만 사용)
        self.calib_lambda = float(args.get('calib_lambda', 0.0))
        self.prior_fg     = float(args.get('prior_fg', 0.015))

        # static 3-class 가중치 (bg=1, small=s_w, large=l_w)
        s_w = float(args.get('s_weights', 30.0))
        l_w = float(args.get('l_weights', 50.0))
        self.register_buffer(
            'static_class_weight',
            torch.tensor([1.0, s_w, l_w], dtype=torch.float32)
        )

        self.loss_dict = {}

    @staticmethod
    def _sanitize_target(target: torch.Tensor, C: int, ignore_index: int):
        """
        - long dtype 보장
        - [0, C-1] 범위를 벗어난 값은 전부 ignore_index로 치환
        - NaN/Inf 방지
        """
        t = torch.nan_to_num(target).long()
        invalid = (t < 0) | (t >= C)
        if invalid.any():
            t = t.clone()
            t[invalid] = ignore_index
        return t

    # --- (선택) 클래스 가중치 길이 보정 ---
    @staticmethod
    def _fix_class_weights(class_weights, C: int, device):
        if class_weights is None:
            return None
        cw = class_weights.to(device=device, dtype=torch.float32)
        if cw.numel() == C:
            return cw
        # 길이 안 맞으면 앞에서 채우고, 모자라면 1.0으로 패딩
        w = torch.ones(C, dtype=torch.float32, device=device)
        n = min(C, cw.numel())
        if n > 0:
            w[:n] = cw[:n]
        return w

    def _stable_ce(self, logits, target, class_weights=None,
                   temperature=1.5, label_smoothing=0.0,
                   ohem_pct=None, ignore_index=255, cap=20.0):

        # ---- 준비: C/타깃 살균/웨이트 보정 ----
        assert logits.dim() == 4, f"expected [N,C,H,W], got {list(logits.shape)}"
        N, C, H, W = logits.shape
        target = self._sanitize_target(target, C=C, ignore_index=ignore_index).contiguous()
        w = self._fix_class_weights(class_weights, C=C, device=logits.device)

        valid = (target != ignore_index)

        # ---- FP32 계산 + temperature ----
        with torch.cuda.amp.autocast(enabled=False):
            lg = (logits.float() / max(1e-6, float(temperature))).contiguous()
            ce_map = F.cross_entropy(
                lg, target,
                weight=w, ignore_index=ignore_index,
                reduction='none',
                label_smoothing=float(label_smoothing)
            )  # [N,H,W] (float32)

        # ---- per-pixel cap & OHEM ----
        ce_valid = torch.clamp(ce_map, max=float(cap))[valid]  # <-- 에러 라인 교체
        if ohem_pct is not None:
            k = max(1, int(float(ohem_pct) * ce_valid.numel()))
            ce_valid, _ = torch.topk(ce_valid, k, sorted=False)
        return ce_valid.mean() if ce_valid.numel() > 0 else ce_map.new_tensor(0.0)

    # (BCE 경로는 이전 버전 그대로면 OK)

    def forward(self, output_dict, gt_dict):
        device = self.static_class_weight.device
        static_loss  = torch.zeros((), device=device)
        dynamic_loss = torch.zeros((), device=device)

        # ----- Dynamic -----
        if self.target in ('dynamic', 'both'):
            dynamic_pred = output_dict['dynamic_seg']        # [B,L,Cd,H,W]
            dynamic_gt   = gt_dict['gt_dynamic']             # [B,L,H,W]
            dyn_logits = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w').contiguous()
            dyn_target = rearrange(dynamic_gt,  'b l h w -> (b l) h w').contiguous()
            Cd = dyn_logits.shape[1]

            if Cd == 2:
                class_weights = torch.tensor([1.0, self.d_weights], dtype=torch.float32, device=dyn_logits.device)
                dynamic_loss = self._stable_ce(
                    dyn_logits, dyn_target,
                    class_weights=class_weights,
                    temperature=self.ce_temperature,
                    label_smoothing=self.ce_label_smoothing,
                    ohem_pct=self.ce_ohem_pct,
                    ignore_index=self.ignore_index,
                    cap=self.ce_cap
                )
            elif Cd == 1:
                # (필요 시 _stable_bce 사용)
                pos_w = torch.tensor([self.d_weights], dtype=torch.float32, device=dyn_logits.device)
                dynamic_loss = self._stable_bce(
                    dyn_logits, dyn_target,
                    pos_weight=pos_w,
                    temperature=self.ce_temperature,
                    ohem_pct=self.ce_ohem_pct,
                    ignore_index=self.ignore_index,
                    cap=self.ce_cap
                )
            else:
                # 예외 케이스: Cd>2면 균등 가중치로 CE
                w_dyn = torch.ones(Cd, dtype=torch.float32, device=dyn_logits.device)
                dynamic_loss = self._stable_ce(
                    dyn_logits, dyn_target,
                    class_weights=w_dyn,
                    temperature=self.ce_temperature,
                    label_smoothing=self.ce_label_smoothing,
                    ohem_pct=self.ce_ohem_pct,
                    ignore_index=self.ignore_index,
                    cap=self.ce_cap
                )

            # calibration (optional)
            if self.calib_lambda > 0:
                with torch.no_grad():
                    valid = (self._sanitize_target(dyn_target, Cd, self.ignore_index) != self.ignore_index)
                p = (torch.softmax(dyn_logits, dim=1)[:, 1] if Cd >= 2
                     else torch.sigmoid(dyn_logits[:, 0]))
                p_valid = p[valid]
                if p_valid.numel() > 0:
                    prior = torch.as_tensor(self.prior_fg, device=p_valid.device, dtype=p_valid.dtype)
                    dynamic_loss = dynamic_loss + self.calib_lambda * (p_valid.mean() - prior).pow(2)

        # ----- Static -----
        if self.target in ('static', 'both'):
            static_pred = output_dict['static_seg']          # [B,L,Cs,H,W]
            static_gt   = gt_dict['gt_static']               # [B,L,H,W]
            sta_logits = rearrange(static_pred, 'b l c h w -> (b l) c h w').contiguous()
            sta_target = rearrange(static_gt,  'b l h w -> (b l) h w').contiguous()
            Cs = sta_logits.shape[1]

            # 현재 Cs에 맞게 weight 생성/보정
            base_w = torch.tensor([1.0, self.s_weights, self.l_weights], dtype=torch.float32, device=sta_logits.device)
            w_static = self._fix_class_weights(base_w, C=Cs, device=sta_logits.device)

            static_loss = self._stable_ce(
                sta_logits, sta_target,
                class_weights=w_static,
                temperature=self.ce_temperature,
                label_smoothing=self.ce_label_smoothing,
                ohem_pct=self.ce_ohem_pct,   # 원하면 static엔 None도 가능
                ignore_index=self.ignore_index,
                cap=self.ce_cap
            )

        # ----- Aux & total -----
        ## position uncertainty 추가 예정
        pos_loss = output_dict.get('pos_loss', [torch.zeros((), device=device)])[0]
        if isinstance(pos_loss, (list, tuple)): pos_loss = pos_loss[0]
        offsets = output_dict.get('offsets', [[torch.zeros((), device=device)]])[0]
        off_loss = (torch.stack([o.abs().mean() if (torch.is_tensor(o) and o.numel() > 0)
                                 else torch.zeros((), device=device) for o in offsets]).mean()
                    if isinstance(offsets, (list, tuple)) else torch.zeros((), device=device))

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss + self.p_coe * pos_loss + self.o_coe * off_loss

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
                            epoch)
            writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                            epoch)
            writer.add_scalar('Position_loss', pos_loss.item(),
                            epoch)
            writer.add_scalar('Offset_loss', offset_loss.item(),
                            epoch)

