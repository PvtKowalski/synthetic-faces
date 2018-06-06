from imgaug import augmenters as iaa
import imgaug as ia

class Augmenter:
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq_geom = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.08, 0.08), "y": (-0.02, 0.02)},
                rotate=(-5, 5),
                shear=(-6, 6),
                order=[0],
                mode='edge'
            )),
            iaa.Sometimes(0.2, iaa.PiecewiseAffine(scale=(0.01, 0.05), order=0))
        ],
        random_order=True
    )
    seq_color = iaa.Sequential(
        [
            iaa.SomeOf((0, 4),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                ],
                random_order=True
            )
        ],
        random_order=True
    )