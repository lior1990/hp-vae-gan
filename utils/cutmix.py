import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_batch(batch, batch_index_permutation):
    lam = np.random.uniform()
    # rand_index = torch.randperm(batch.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(), lam)
    batch[:, :, bbx1:bbx2, bby1:bby2] = batch[batch_index_permutation, :, bbx1:bbx2, bby1:bby2]
