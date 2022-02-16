import unittest
from imap.model.nerf import NERF
from imap.utils import UniversalFactory
from pytorch_lightning.utilities.parsing import AttributeDict
from imap.model.embeddings.gaussian_positional_encoding import GaussianPositionalEncoding
from imap.model.embeddings.gaussian_positional_embedding import GaussianPositionalEmbedding
from imap.data.datasets.tum.tum_dataset_loader_factory import TUMDatasetLoaderFactory
import torch


class TestStratifiedDepthSampling(unittest.TestCase):

    def setUp(self) -> None:
        dataset_params = {'dataset_path': "./datasets/tum rgbd/",
                          'scene_name': "rgbd_dataset_freiburg1_desk",
                          'association_file_name': "data_association_file.txt",
                          'frame_indices': [407],
                          'distance_koef': 0.1,
                          'clip_distance_threshold': 4.}
        camera, dataset_loader = TUMDatasetLoaderFactory.make_dataset_loader(**dataset_params)

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
        self.model = factory.make_from_parameters(model_parameters, camera_info=camera)
        self.batch_size = 5
        self.device = 'cpu'

    def test_deterministic_sampling(self):
        depths = self.model.stratified_sample_depths(batch_size=self.batch_size,
                                                     device=self.device,
                                                     bins_count=self.model.fine_sample_bins,
                                                     deterministic=True)
        self.assertTrue(depths.shape == (self.model.fine_sample_bins, self.batch_size))
        self.assertTrue(torch.all(depths[0] == self.model.minimal_depth))
        self.assertTrue(torch.all(depths <= self.model._default_depth))

        delta = (self.model._default_depth.item() - self.model.minimal_depth) / self.model.fine_sample_bins
        self.assertTrue(torch.all((depths[-1] - depths[-2]) <= delta + 1e-5))

    def test_random_sampling(self):
        depths = self.model.stratified_sample_depths(batch_size=self.batch_size,
                                                     device=self.device,
                                                     bins_count=self.model.fine_sample_bins,
                                                     deterministic=False)
        self.assertTrue(depths.shape == (self.model.fine_sample_bins, self.batch_size))
        self.assertTrue(torch.all(depths[0] == self.model.minimal_depth))
        self.assertTrue(torch.all(depths <= self.model._default_depth))
