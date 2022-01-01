from typing import Union, Tuple
import numpy as np
import SimpleITK as sitk

# U: Uniformly sample
def U(umin, umax):
    return np.random.choice(np.linspace(umin, umax, 100)).item()

class RegistrationSimulator3D:
    """
    Parameters from the original paper are the default values.
    The elastic deformation is using bspline registration.
    Note: the rotation parameter is in degrees.
    Note: this class does not save transforms
    """

    def __init__(
        self,
        reference_image: sitk.Image,
        spatial_shape: Tuple[int, int, int],
        rotation_min: Union[float, Tuple[float, float, float]] = 0.0,
        rotation_max: Union[float, Tuple[float, float, float]] = 30.0,
        scale_min: Union[float, Tuple[float, float, float]] = 0.75,
        scale_max: Union[float, Tuple[float, float, float]] = 1.25,
        translation_factor: Union[float, Tuple[float, float, float]] = 0.02,
        offset_gaussian_std_max: int = 1000,
        smoothing_gaussian_std_min: int = 10,
        smoothing_gaussian_std_max: int = 13,
    ):

        super().__init__()
        self.reference_image = reference_image
        self.ndims = 3
        self.spatial_shape = spatial_shape
        self.rotation_min = self.__tuplify(rotation_min)
        self.rotation_max = self.__tuplify(rotation_max)
        self.scale_min = self.__tuplify(scale_min)
        self.scale_max = self.__tuplify(scale_max)
        self.translation_factor = self.__tuplify(translation_factor)

        self.offset_gaussian_std_max = offset_gaussian_std_max
        self.smoothing_gaussian_std_min = smoothing_gaussian_std_min
        self.smoothing_gaussian_std_max = smoothing_gaussian_std_max
        self.offset_gaussian_std_max = offset_gaussian_std_max

        self.center = [(s - 1) / 2 for s in spatial_shape]

    def __tuplify(
        self, x: Union[float, Tuple[float, ...]]
    ) -> Tuple[float, float, float]:
        """
        A necessary evil to get around indeterminate Tuple length errors in Pyright.
        Solves the problem of:
            assert len(some_list) == 3
            x: Tuple[t,t,t] = tuple(some_list) <- Pyright throws an error here
        """
        if isinstance(x, tuple):
            assert len(x) == 3
            return (x[0], x[1], x[2])
        return (x, x, x)

    def __get_scale_transform(self) -> sitk.ScaleTransform:
        scales = self.__tuplify(
            tuple([U(smin, smax) for smin, smax in zip(self.scale_min, self.scale_max)])
        )
        transform = sitk.ScaleTransform(self.ndims)
        transform.SetScale(np.asarray(scales))

        return transform

    def __get_rot_trans_transform(self) -> sitk.Euler3DTransform:
        rotations = self.__tuplify(
            tuple(
                [
                    U(rmin, rmax)
                    for rmin, rmax in zip(self.rotation_min, self.rotation_max)
                ]
            )
        )

        radians = np.deg2rad(rotations)

        tfs = [U(-i, i) for i in self.translation_factor]
        translation = self.__tuplify(
            tuple([U(0, self.spatial_shape[i] * tf) for i, tf in enumerate(tfs)])
        )

        transform = sitk.Euler3DTransform()
        transform.SetTranslation(translation)
        transform.SetCenter(tuple(self.center))
        transform.SetRotation(*radians)

        return transform

    def __get_elastic_transform(self) -> sitk.DisplacementFieldTransform:
        offset_std = U(0, self.offset_gaussian_std_max)
        smoothed_offset_field = np.zeros((self.ndims, *self.spatial_shape))
        smoothing_std = U(
            self.smoothing_gaussian_std_min, self.smoothing_gaussian_std_max
        )
        smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        smoothing_filter.SetSigma(smoothing_std)
        for i in range(self.ndims):
            offset_i = np.random.normal(0, offset_std, size=self.spatial_shape)
            offset_sitk = sitk.GetImageFromArray(offset_i, isVector=False)
            smoothed_offset_i = smoothing_filter.Execute(offset_sitk)
            smoothed_offset_field[i] = sitk.GetArrayFromImage(smoothed_offset_i)

        smoothing_filter.Execute

        smoothed_offset_sitk = sitk.GetImageFromArray(
            smoothed_offset_field.T, isVector=True
        )

        elastic_distortion = sitk.DisplacementFieldTransform(smoothed_offset_sitk)
        return elastic_distortion

    def generate_random_transform(self) -> sitk.CompositeTransform:
        """
        Generates new transforms
        """

        transforms = [
            self.__get_scale_transform(),
            self.__get_rot_trans_transform(),
            self.__get_elastic_transform(),
        ]

        # SITK transforms are applied in reverse order, so the list needs to be reversed
        composite = sitk.CompositeTransform(transforms[::-1])
        return composite

    def get_random_displacement_field(self) -> sitk.sitkDisplacementField:
        """
        Generates new displacement field filter
        """
        transform = self.generate_random_transform()
        t2df = sitk.TransformToDisplacementFieldFilter()
        t2df.SetReferenceImage(self.reference_image)
        displacement_field = t2df.Execute(transform)
        return displacement_field

    def get_displacement_array(self) -> np.ndarray:
        df = self.get_random_displacement_field()
        # Transpose df to get dimension (3) into the first axis
        return sitk.GetArrayFromImage(df).T
   