import unittest
from imap.model.nerf import NERF
from imap.utils import UniversalFactory
from pytorch_lightning.utilities.parsing import AttributeDict
from imap.utils.torch_math import back_project_pixel
from imap.model.embeddings.gaussian_positional_embedding import GaussianPositionalEmbedding
from imap.data.datasets.tum.tum_dataset_loader_factory import TUMDatasetLoaderFactory
from imap.model.active_sampling.image_active_sampling import ImageActiveSampling
from imap.trainers.trainers import ModelTrainer
import torch


class TestBackprojection(unittest.TestCase):
    def setUp(self) -> None:
        dataset_params = {'dataset_path': "./datasets/tum rgbd/",
                          'scene_name': "rgbd_dataset_freiburg1_desk",
                          'association_file_name': "data_association_file.txt",
                          'frame_indices': [407],
                          'distance_koef': 0.1,
                          'clip_distance_threshold': 4.}
        camera, self.dataset_loader = TUMDatasetLoaderFactory.make_dataset_loader(**dataset_params)

        model_parameters = AttributeDict(
            name="NERF",
            course_sample_bins=12,
            fine_sample_bins=10,
            depth_loss_koef=1.,
            color_loss_koef=5.,
            minimal_depth=0.01,
            positional_embedding=AttributeDict(
                name='GaussianPositionalEmbedding',
                encoding_dimension=93,
                sigma=25,
                use_only_sin=False,
                use_bias=True
            )
        )
        factory = UniversalFactory([NERF, GaussianPositionalEmbedding])
        self.batch_size = 200
        self.device = 'cpu'
        self.model = factory.make_from_parameters(model_parameters, camera_info=camera).to(self.device)
        image_active_sampler = ImageActiveSampling(camera, points_per_frame=self.batch_size)
        self.trainer = ModelTrainer(image_active_sampler)

    def test_backprojection(self):
        for state in self.dataset_loader:
            data_batch = self.trainer.sample_batch(state, state.frame.get_pixel_probs())
            self.trainer.send_batch_to_model_device(data_batch, 'cpu')
            pixels = data_batch['pixel']
            sampled_depths = self.model.stratified_sample_depths(pixels.shape[0],
                                                                 pixels.device,
                                                                 self.model.course_sample_bins,
                                                                 deterministic=False)
            depths = torch.sort(sampled_depths, dim=0).values
            back_projected_points = back_project_pixel(pixels, depths, data_batch['camera_position'],
                                                       self.model._inverted_camera_matrix)
            self.assertTrue(torch.all(back_projected_points < 1))
            self.assertTrue(torch.all(back_projected_points > -1))
            bins_count = sampled_depths.shape[0]
            self.assertTrue(back_projected_points.shape == (self.batch_size * bins_count, 3))
