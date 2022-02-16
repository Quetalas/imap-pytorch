class Predict:
    def __init__(self,
                 fine_color,
                 fine_depths,
                 fine_depth_variance,
                 coarse_color=None,
                 coarse_depths=None,
                 coarse_depth_variance=None):
        self.fine_color = fine_color
        self.fine_depths = fine_depths
        self.fine_depth_variance = fine_depth_variance
        self.coarse_color = coarse_color
        self.coarse_depths = coarse_depths
        self.coarse_depth_variance = coarse_depth_variance

    def to_numpy_cpu(self):
        self.fine_color = self.fine_color.detach().cpu().numpy()
        self.fine_depths = self.fine_depths.detach().cpu().numpy()
        self.fine_depth_variance = self.fine_depth_variance.detach().cpu().numpy()
        self.coarse_color = self.coarse_color.detach().cpu().numpy()
        self.coarse_depths = self.coarse_depths.detach().cpu().numpy()
        self.coarse_depth_variance = self.coarse_depth_variance.detach().cpu().numpy()
        return self
