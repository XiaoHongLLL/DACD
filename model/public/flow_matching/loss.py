import torch
import torch.nn as nn


class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, v_pred, u_t, mask=None):
        pass
        loss = self.mse(v_pred, u_t)

        if mask is not None:
            loss = loss * mask

            return loss.sum() / mask.sum()
        else:
            return loss.mean()
