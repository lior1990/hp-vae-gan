import torch


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, opt, cls_batch):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(opt.device)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(pad_with_cls(interpolates, cls_batch, opt))

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def pad_with_cls(batch, cls_batch, opt):
    cls = torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 1, device=opt.device) * cls_batch.view(-1, 1, 1, 1)
    return torch.cat([batch, cls], dim=1)
