import random
from torchvision.transforms import functional as TFF
from torchvision import transforms as TF

def video_paired_transform(
    frames1, 
    frames2, 
    crop_size=128, 
    hflip=True, 
    rotation=True,
    crop_prob=0.7, 
):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    # -------- Random decision: crop or resize --------
    use_crop = random.random() < crop_prob
    if use_crop:
        # Check if image is large enough, otherwise force resize
        if frames1[0].height < crop_size[0] or frames1[0].width < crop_size[1]:
            use_crop = False

    # -------- Random augmentation parameters --------
    do_hflip = hflip and random.random() < 0.5
    do_vflip = rotation and random.random() < 0.5
    do_rot90 = rotation and random.random() < 0.5

    if use_crop:
        i, j, h, w = TF.RandomCrop.get_params(frames1[0], output_size=crop_size)
    
    # -------- Unified transformation function --------
    def apply(frames):
        # Strategy branch
        if use_crop:
            frames = [TF.crop(frame, i, j, h, w) for frame in frames]
        else:
            frames = [TF.resize(frame, crop_size) for frame in frames]  # direct resize original image
        
        # Common augmentations
        if do_hflip:
            frames = [TF.hflip(frame) for frame in frames]
        if do_vflip:
            frames = [TF.vflip(frame) for frame in frames]
        if do_rot90:
            frames = [TF.rotate(frame, 90) for frame in frames]

        frames = [TF.to_tensor(frame) for frame in frames]
        return frames

    # Apply transform
    frames1 = apply(frames1)
    frames2 = apply(frames2)
    return frames1, frames2

def paired_transform(
    img1, 
    img2,
    mask=None, 
    crop_size=128, 
    hflip=True, 
    rotation=True,
    crop_prob=0.7, 
):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    # -------- Random decision: crop or resize --------
    use_crop = random.random() < crop_prob
    if use_crop:
        if img1.height < crop_size[0] or img1.width < crop_size[1]:
            use_crop = False

    # -------- Random augmentation parameters --------
    do_hflip = hflip and random.random() < 0.5
    do_vflip = rotation and random.random() < 0.5
    do_rot90 = rotation and random.random() < 0.5

    # -------- If cropping, generate shared crop params --------
    if use_crop:
        i, j, h, w = TF.RandomCrop.get_params(img1, output_size=crop_size)

    # -------- Common transform function --------
    def apply(img):
        if use_crop:
            img = TFF.crop(img, i, j, h, w)
        else:
            img = TFF.resize(img, crop_size)
        if do_hflip:
            img = TFF.hflip(img)
        if do_vflip:
            img = TFF.vflip(img)
        if do_rot90:
            img = TFF.rotate(img, 90)
        return TFF.to_tensor(img)

    img1 = apply(img1)
    img2 = apply(img2)
    if mask is not None:
        mask = TFF.rgb_to_grayscale(apply(mask))
        return img1, img2, mask
    return img1, img2