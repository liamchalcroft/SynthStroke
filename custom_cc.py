import torch
import cornucopia as cc



class RandomGaussianMixtureTransform(torch.nn.Module):
    """Sample from a Gaussian mixture with known cluster assignment"""

    def __init__(self, mu=255, sigma=16, fwhm=2, background=None,
                 dtype=None):
        """
        Parameters
        ----------
        mu : list[float]
            Mean of each cluster
        sigma : list[float]
            Standard deviation of each cluster
        fwhm : float or list[float], optional
            Width of a within-class smoothing kernel.
        background : int, optional
            Index of background channel
        dtype : torch.dtype
            Output data type. Only used if input is an integer label map.
        """
        super().__init__()
        self.dtype = dtype
        self.sample = dict(mu=cc.random.Uniform.make(cc.random.make_range(0, mu)),
                      sigma=cc.random.Uniform.make(cc.random.make_range(0, sigma)),
                      fwhm=cc.random.Uniform.make(cc.random.make_range(0, fwhm)))
        self.background = background

    def forward(self, x):
        theta = self.get_parameters(x)
        return self.apply_transform(x, theta)

    def get_parameters(self, x):
        mu = self.sample['mu'](len(x))
        sigma = self.sample['sigma'](len(x))
        fwhm = int(self.sample['fwhm']())
        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
        else:
            backend = dict(dtype=self.dtype or torch.get_default_dtype(),
                           device=x.device)
        mu = torch.as_tensor(mu).to(**backend)
        sigma = torch.as_tensor(sigma).to(**backend)
        return mu, sigma, fwhm

    def apply_transform(self, x, parameters):
        mu, sigma, fwhm = parameters

        backend = dict(dtype=x.dtype, device=x.device)
        y = torch.zeros_like(x[0], **backend)
        mu = mu.to(**backend)
        sigma = sigma.to(**backend)
        if self.background is not None:
            x[self.background] = 0
        y1 = torch.randn(*x.shape, **backend)
        y1 = cc.utils.conv.smoothnd(y1, fwhm=fwhm)
        y1 = y1.mul_(sigma[...,None,None,None]).add_(mu[...,None,None,None])

        y = torch.sum(x * y1, dim=0)
        y = y[None]
        return y
    

class SynthFromLabelTransform(cc.Transform):
    """
    Synthesize an MRI from an existing label map
    Examples
    --------
    ::
        # if inputs are preloaded label tensors (default)
        synth = SynthFromLabelTransform()
        # if inputs are filenames
        synth = SynthFromLabelTransform(from_disk=True)
        # memory-efficient patch-synthesis
        synth = SynthFromLabelTransform(patch=64)
        img, lab = synth(input)
    References
    ----------
    ..[1] "SynthSeg: Domain Randomisation for Segmentation of Brain
          MRI Scans of any Contrast and Resolution"
          Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher,
          Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias
          2021
          https://arxiv.org/abs/2107.09559
    """

    def __init__(self,
                 num_ch=1,
                 patch=None,
                 rotation=15,
                 shears=0.012,
                 zooms=0.15,
                 elastic=0.05,
                 elastic_nodes=10,
                 gmm_fwhm=10,
                 bias=7,
                 gamma=0.6,
                 motion_fwhm=3,
                 resolution=8,
                 snr=10,
                 gfactor=5,
                 order=3,
                 skip_gmm=False,):
        """
        Parameters
        ----------
        patch : [list of] int, optional
            Shape of the patches to extact
        from_disk : bool, default=False
            Assume inputs are filenames and load from disk
        one_hot : bool, default=False
            Return one-hot labels. Else return a label map.
        synth_labels : tuple of [tuple of] int, optional
            List of labels to use for synthesis.
            If multiple labels are grouped in a sublist, they share the
            same intensity in the GMM. All labels not listed are assumed
            background.
        synth_labels_maybe : dict(tuple of [tuple of] int -> float), optional
            List of labels to sometimes use for synthesis, and their
            probability of being sampled.
        target_labels : tuple of [tuple of] int, optional
            List of target labels.
            If multiple labels are grouped in a sublist, they are fused.
            All labels not listed are assumed background.
            The final label map is relabeled in the order provided,
            starting from 1 (background is 0).
        order : int
            Spline order of the elastic and bias fields (1 is much faster)
        Geometric Parameters
        --------------------
        rotation : float
            Upper bound for rotations, in degree.
        shears : float
            Upper bound for shears
        zooms : float
            Upper bound for zooms (about one)
        elastic : float
            Upper bound for elastic displacements, in percent of the FOV.
        elastic_nodes : int
            Upper bound for number of control points in the elastic field.
        Intensity Parameters
        --------------------
        gmm_fwhm : float
            Upper bound for the FWHM of the intra-tissue smoothing kernel
        bias : int
            Upper bound for the number of control points of the bias field
        gamma : float
            Upper bound for the exponent of the Gamma transform
        motion_fwhm : float
            Upper bound of the FWHM of the global (PSF/motion) smoothing kernel
        resolution : float
            Upper bound for the inter-slice spacing (in voxels)
        snr : float
            Lower bound for the signal-to-noise ratio
        gfactor : int
            Upper bound for the number of control points of the g-factor map
        """
        super().__init__(shared=True)
        self.deform = cc.RandomAffineElasticTransform(
            elastic, elastic_nodes, order=order, bound='zeros',
            rotations=rotation, shears=shears, zooms=zooms, patch=patch)
        self.gmm = RandomGaussianMixtureTransform(fwhm=gmm_fwhm, background=0) if skip_gmm is not True else None
        self.intensity = cc.IntensityTransform(
            bias, gamma, motion_fwhm, resolution, snr, gfactor, order)
        self.num_ch = num_ch

    def get_parameters(self, x):
        parameters = dict()
        parameters['gmm'] = [self.gmm.get_parameters(x) for i in range(self.num_ch)]
        parameters['deform'] = self.deform.get_parameters(x)
        return parameters

    def forward(self, x, coreg=None):
        theta = self.get_parameters(x)
        return self.apply_transform(x, theta, coreg)

    def apply_transform(self, lab, parameters=None, coreg=None):
        # use coreg for any other labels/images that must be coregistered to synth image
        lab = self.deform.apply_transform(lab, parameters['deform'])
        if self.gmm is not None:
            img = torch.cat([self.intensity(self.gmm.apply_transform(lab, parameters['gmm'][i])) for i in range(self.num_ch)], dim=0)
        else:
            img = self.intensity(lab)
        if coreg is not None:
            if isinstance(coreg, (list, tuple)):
                coreg = [self.deform.apply_transform(cor, parameters['deform']) for cor in coreg]
            else:
                coreg = self.deform.apply_transform(coreg, parameters['deform'])
            return img, lab, coreg
        return img, lab
    

class CCSynthSeg:
    """ """


    def __init__(
        self,
        label_key,
        image_key='image',
        coreg_keys=None,
        num_ch=1,
        patch=None, # currently not working how we want - generates random crops for all coregs rather than uniform crop
        rotation=15,
        shears=0.012,
        zooms=0.15,
        elastic=0.05,
        elastic_nodes=10,
        gmm_fwhm=10,
        bias=7,
        gamma=0.6,
        motion_fwhm=3,
        resolution=8,
        snr=10,
        gfactor=5,
        order=3,
        skip_gmm=False
    ) -> None:
        self.label_key = label_key
        self.image_key = image_key
        self.coreg_keys = coreg_keys if isinstance(coreg_keys, (tuple, list)) else [coreg_keys]
        self.transform = SynthFromLabelTransform(
            num_ch=num_ch,
            patch=patch,
            rotation=rotation,
            shears=shears,
            zooms=zooms,
            elastic=elastic,
            elastic_nodes=elastic_nodes,
            gmm_fwhm=gmm_fwhm,
            bias=bias,
            gamma=gamma,
            motion_fwhm=motion_fwhm,
            resolution=resolution,
            snr=snr,
            gfactor=gfactor,
            order=order,
            skip_gmm=False
        )

    def __call__(
        self, data
    ):
        d = dict(data)
        d[self.image_key], d[self.label_key], coreg = self.transform(d[self.label_key], [d[key] for key in self.coreg_keys] if self.coreg_keys is not None else None)
        if self.label_key + "_meta_dict" in list(d.keys()):
            d[self.image_key+"_meta_dict"] = d[self.label_key+"_meta_dict"]
        if coreg is not None:
            for i, key in enumerate(self.coreg_keys):
                d[key] = coreg[i]
        return d