import torch
import torch.nn as nn

from imap.model.implicit_representations.mlp import MLP
from ..utils.torch_math import back_project_pixel
from imap.model.model_predict import Predict
from imap.model.model_loss import ModelLoss


class NERF(nn.Module):
    def __init__(self,
                 positional_embedding,
                 course_sample_bins,
                 fine_sample_bins,
                 depth_loss_koef,
                 color_loss_koef,
                 minimal_depth,
                 camera_info
                 ):
        super().__init__()
        assert course_sample_bins > 0 and fine_sample_bins > 0
        self.course_sample_bins = course_sample_bins
        self.fine_sample_bins = fine_sample_bins
        self.depth_loss_koef = depth_loss_koef
        self.color_loss_koef = color_loss_koef
        self.minimal_depth = minimal_depth

        self._positional_encoding = positional_embedding
        self._mlp = MLP(self._positional_encoding.encoding_dimension, 4)

        self._inverted_camera_matrix = torch.tensor(camera_info.get_inverted_camera_matrix(), requires_grad=False)
        self._default_color = torch.tensor(camera_info.get_default_color(), requires_grad=False)
        self._default_depth = torch.tensor(camera_info.get_default_depth(), requires_grad=False)
        self._loss = nn.L1Loss(reduction="none")

    def forward(self, pixel, camera_position):
        """
        :param pixel: (batch_size, 2), [x, y]
        :param camera_position: [[R.t 0],
                                [T 1]]
                                supposed that R is transposed for compatability with pytorch3d transformations
        :return:
        """
        with torch.no_grad():
            course_sampled_depths = self.stratified_sample_depths(
                pixel.shape[0],
                pixel.device,
                self.course_sample_bins,
                not self.training)
        coarse_color, coarse_depths, coarse_weights, coarse_depth_variance = self.reconstruct_color_and_depths(
            course_sampled_depths,
            pixel,
            camera_position.T,
            self._mlp)
        with torch.no_grad():
            fine_sampled_depths = self.hierarchical_sample_depths(
                coarse_weights,
                pixel.shape[0],
                pixel.device,
                self.fine_sample_bins,
                course_sampled_depths,
                not self.training)
        fine_sampled_depths = torch.cat([fine_sampled_depths, course_sampled_depths], dim=0)
        # fine_sampled_depths = torch.cat([course_sampled_depths], dim=0)
        fine_color, fine_depths, fine_weights, fine_depth_variance = self.reconstruct_color_and_depths(
            fine_sampled_depths,
            pixel,
            camera_position.T,
            self._mlp)
        return Predict(fine_color,
                       fine_depths,
                       fine_depth_variance,
                       coarse_color,
                       coarse_depths,
                       coarse_depth_variance)

    def reconstruct_color_and_depths(self, sampled_depths, pixels, camera_positions, mlp_model):
        """
        :param sampled_depths:
        :param pixels:
        :param camera_positions: [[R T],
                                  [0 1]]
        :param mlp_model:
        :return:
        """
        bins_count = sampled_depths.shape[0]
        depths = torch.sort(sampled_depths, dim=0).values
        back_projected_points = back_project_pixel(pixels, depths, camera_positions,
                                                   self._inverted_camera_matrix)
        colors, density = self.forward_network(back_projected_points, mlp_model)
        colors = colors.reshape(bins_count, -1, 3)
        density = density.reshape(bins_count, -1)
        weights = self.calculate_weights(density, depths)

        reconstructed_color = self.reconstruct_color(colors, weights, self._default_color)
        reconstructed_depths = self.reconstruct_depth(depths, weights, self._default_depth)
        with torch.no_grad():
            reconstructed_depth_variance = self.reconstruct_depth_variance(depths, weights, reconstructed_depths,
                                                                           self._default_depth)
        return reconstructed_color, reconstructed_depths, weights, reconstructed_depth_variance

    def forward_network(self, points, mlp_model):
        """
        :param points: (batch_size, 3)
        :param mlp_model:
        :return: colors
        """
        encodings = self._positional_encoding(points)
        prediction = mlp_model(encodings)
        colors = torch.sigmoid(prediction[:, :3])
        density = torch.relu(prediction[:, 3])
        return colors, density

    def stratified_sample_depths(self, batch_size, device, bins_count, deterministic=False):
        """
        :param batch_size: int
        :param device
        :param bins_count: int
        :param deterministic: bool
        :return: (bins_count, batch_size)
            result[0] - the closest depths
            result[-1] - the farthest depths
        """
        if deterministic:
            depth_delta = (self._default_depth.item() - self.minimal_depth) / bins_count
            result = torch.arange(self.minimal_depth, self._default_depth.item(), depth_delta, device=device,
                                  requires_grad=False)
            result = torch.repeat_interleave(result[:, None], batch_size, dim=1)
            return result
        uniform = torch.rand((bins_count, batch_size), device=device, requires_grad=False)
        uniform[0] = 1
        result = (torch.arange(bins_count, device=device, requires_grad=False)[:, None] + uniform - 1
                  ) * (self._default_depth - self.minimal_depth) / (bins_count - 1) + self.minimal_depth
        return result

    def hierarchical_sample_depths(self, weights, batch_size, device, bins_count, bins, deterministic=False):
        weights = weights.transpose(1, 0)[:, :-1] + 1e-10
        pdf = weights / torch.sum(weights, dim=1)[:, None]
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], 1)
        minimal_bin = bins[0]
        bins = (torch.roll(bins, 1) + bins) / 2
        bins[0] = minimal_bin
        bins = bins.transpose(1, 0)

        if deterministic:
            uniform = torch.arange(bins_count, device=device) / bins_count + 1. / 2 / bins_count
            uniform = torch.repeat_interleave(uniform[None], batch_size, dim=0)
        else:
            uniform = torch.rand((batch_size, bins_count), device=device).contiguous()
        indexes = torch.searchsorted(cdf, uniform, right=True)
        index_below = self.clip_indexes(indexes - 1, 0, bins.shape[1] - 1)
        index_above = self.clip_indexes(indexes, 0, bins.shape[1] - 1)

        denominator = torch.gather(cdf, 1, index_above) - torch.gather(cdf, 1, index_below)
        denominator = torch.where(denominator < 1e-10, torch.ones_like(denominator), denominator)
        t = (uniform - torch.gather(cdf, 1, index_below)) / denominator
        bins_below = torch.gather(bins, 1, index_below)
        bins_above = torch.gather(bins, 1, index_above)
        hierarchical_sample = bins_below + t * (bins_above - bins_below)
        return hierarchical_sample.transpose(1, 0)

    @staticmethod
    def clip_indexes(indexes, minimal, maximal):
        result = torch.max(minimal * torch.ones_like(indexes), indexes)
        result = torch.min(maximal * torch.ones_like(indexes), result)
        return result

    def calculate_weights(self, densities, depths):
        weights = []
        product = 1
        for i in range(len(depths)):
            if i < len(depths) - 1:
                depth_delta = depths[i + 1] - depths[i]
            else:
                depth_delta = self._default_depth - depths[i]
            hit_probability = 1 - torch.exp(-densities[i] * depth_delta)
            weights.append(hit_probability * product)
            product = product * (1 - hit_probability)
        weights.append(product)
        return torch.stack(weights, dim=0)

    @staticmethod
    def reconstruct_color(colors, weights, default_color):
        return torch.sum(colors * weights[:-1, :, None], dim=0
                         ) + default_color.to(colors.device)[None] * weights[-1, :, None]

    @staticmethod
    def reconstruct_depth(depths, weights, default_depth):
        return torch.sum(depths * weights[:-1, :], dim=0) + default_depth.to(depths.device)[None] * weights[-1]

    @staticmethod
    def reconstruct_depth_variance(depths, weights, mean_depths, default_depth):
        return torch.sum((depths - mean_depths[None]) ** 2 * weights[:-1], dim=0
                         ) + (default_depth.to(depths.device)[None] - mean_depths) ** 2 * weights[-1]

    def photometric_loss(self, rendered_colors, true_colors):
        return torch.mean(self._loss(rendered_colors, true_colors), dim=1)

    def geometric_loss(self, rendered_depths, true_depths):
        return self._loss(rendered_depths, true_depths)

    def normalized_geometric_loss(self, geometric_loss, depth_variance):
        return geometric_loss / depth_variance

    def losses(self, predict, true_colors, true_depths):
        """
        Return photometric & geometric losses for fine & coarse reconstructions
        :param predict: Predict
        :param true_colors:
        :param true_depths:
        :return: { "key": array}    array.shape == torch.size([num_points])
        """
        coarse_image_loss = self.photometric_loss(predict.coarse_color, true_colors)
        fine_image_loss = self.photometric_loss(predict.fine_color, true_colors)

        coarse_depth_loss = self.normalized_geometric_loss(
            self.geometric_loss(predict.coarse_depths, true_depths),
            torch.sqrt(predict.coarse_depth_variance) + 1e-10
        )
        fine_depth_loss = self.normalized_geometric_loss(
            self.geometric_loss(predict.fine_depths, true_depths),
            torch.sqrt(predict.fine_depth_variance) + 1e-10
        )
        # image_loss = course_image_loss + fine_image_loss
        # depth_loss = coarse_depth_loss + fine_depth_loss
        image_loss = fine_image_loss
        depth_loss = fine_depth_loss
        loss = self.color_loss_koef * image_loss + self.depth_loss_koef * depth_loss
        losses = ModelLoss(coarse_image_loss, coarse_depth_loss, fine_image_loss, fine_depth_loss, loss)
        return losses

    def mean_loss(self, loss):
        return torch.mean(loss)
