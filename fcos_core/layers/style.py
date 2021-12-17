
 
import torch
import torch.nn.functional as F
from torch import nn


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C, H * W]), featmap_tgt_S.view([B, C, H * W])
        # calculate Gram matrices
        A_src, A_tgt = torch.bmm(f_src, f_src.transpose(1, 2)), torch.bmm(f_tgt, f_tgt.transpose(1, 2))
        A_src, A_tgt = A_src / (H * W), A_tgt / (H * W)
        loss = F.mse_loss(A_src, A_tgt)
        return loss

class ChannelSimLoss(nn.Module):
    def __init__(self):
        super(ChannelSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        loss = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src @ f_src.t(), f_tgt @ f_tgt.t()
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss += torch.norm(A_src - A_tgt) ** 2 / C
            # loss += torch.norm(A_src - A_tgt, p=1)
        loss /= B
        return loss
        
class BatchSimLoss(nn.Module):
    def __init__(self):
        super(BatchSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C * H * W]), featmap_tgt_S.view([B, C * H * W])
        A_src, A_tgt = f_src @ f_src.t(), f_tgt @ f_tgt.t()
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_batch = torch.norm(A_src - A_tgt) ** 2 / B
        return loss_batch


class PixelSimLoss(nn.Module):
    def __init__(self):
        super(PixelSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        loss_pixel = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src.t() @ f_src, f_tgt.t() @ f_tgt
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_pixel += torch.norm(A_src - A_tgt) ** 2 / (H * W)
        loss_pixel /= B
        return loss_pixel

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        # COMPUTE total variation regularization loss
        loss_var_l2 = ((image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2).mean() + \
                      ((image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2).mean()

        loss_var_l1 = ((image[:, :, :, 1:] - image[:, :, :, :-1]).abs()).mean() + \
                      ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs()).mean()
        return loss_var_l1, loss_var_l2

class KDLoss(nn.Module):
    def __init__(self, temperature=1):
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        """
        NOTE: the KL Divergence for PyTorch comparing the prob of teacher and log prob of student,
        mimicking the prob of ground truth (one-hot) and log prob of network in CE loss
        """
        # x -> input -> log(q)
        log_q = F.log_softmax(student_output / self.temperature, dim=1)
        # y -> target -> p
        p = F.softmax(teacher_output / self.temperature, dim=1)
        # F.kl_div(x, y) -> F.kl_div(log_q, p)
        # l_n = y_n \cdot \left( \log y_n - x_n \right) = p * log(p/q)
        l_kl = F.kl_div(log_q, p)  # forward KL
        return l_kl