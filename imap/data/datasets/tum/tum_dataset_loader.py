class TUMDatasetLoader(object):
    def __init__(self, camera, color_images, depth_images, positions):
        """
        :param color_images:
        :param depth_images:
        :param positions:
        """
        self._color_image = color_images
        self._depth_image = depth_images
        self._positions = positions
        self.camera = camera

    def __getitem__(self, index):
        return self.camera.create_state(self._color_image[index],
                                        self._depth_image[index],
                                        self._positions[index])

    def __len__(self):
        return len(self._color_image)
