import torch


class ModelLoss:
    def __init__(self,
                 coarse_image_loss,
                 coarse_depth_loss,
                 fine_image_loss,
                 fine_depth_loss,
                 loss):
        self.coarse_image_loss = coarse_image_loss
        self.coarse_depth_loss = coarse_depth_loss
        self.fine_image_loss = fine_image_loss
        self.fine_depth_loss = fine_depth_loss
        self.loss = loss

    def mean_loss(self):
        return ModelLoss(torch.mean(self.coarse_image_loss),
                         torch.mean(self.coarse_depth_loss),
                         torch.mean(self.fine_image_loss),
                         torch.mean(self.fine_depth_loss),
                         torch.mean(self.loss))

    def __add__(self, other):
        return ModelLoss(self.coarse_image_loss + other.coarse_image_loss,
                         self.coarse_depth_loss + other.coarse_depth_loss,
                         self.fine_image_loss + other.fine_image_loss,
                         self.fine_depth_loss + other.fine_depth_loss,
                         self.loss + other.loss)

    def divide(self, divider):
        self.coarse_image_loss = torch.div(self.coarse_image_loss, divider)
        self.coarse_depth_loss = torch.div(self.coarse_depth_loss, divider)
        self.fine_image_loss = torch.div(self.fine_image_loss, divider)
        self.fine_depth_loss = torch.div(self.fine_depth_loss, divider)
        self.loss = torch.div(self.loss, divider)

    def detach(self):
        return ModelLoss(self.coarse_image_loss.detach(),
                         self.coarse_depth_loss.detach(),
                         self.fine_image_loss.detach(),
                         self.fine_depth_loss.detach(),
                         self.loss.detach())

    def item(self):
        return ModelLoss(self.coarse_image_loss.item(),
                         self.coarse_depth_loss.item(),
                         self.fine_image_loss.item(),
                         self.fine_depth_loss.item(),
                         self.loss.item())

    def __lt__(self, other):
        return self.loss < other.loss
