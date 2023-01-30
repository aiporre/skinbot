import logging
import sys
import os
import torch
import pathlib


def validate_target_mode(_target_mode, comparable_items):
    if any([item in _target_mode.lower() for item in comparable_items]):
        return True
    else:
        return False

def get_log_path(config):
    logger_dir = config['LOGGER']['logfilepath']
    logger_fname = config['LOGGER']['logfilename']
    return os.path.join(logger_dir, logger_fname)

def configure_logging(config):
    # configure the logging system
    log_level = config['LOGGER']['loglevel']
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    if config['LOGGER']['logtofile'] == 'True':
        logger_dir = config['LOGGER']['logfilepath']
        logger_fname = config['LOGGER']['logfilename']
        logger_path = os.path.join(logger_dir, logger_fname)
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        print('Logging to: ', logger_path, ' with level: ', log_level)
        logging.basicConfig(filename=logger_path, filemode='w', level=numeric_level)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.basicConfig(level=numeric_level, handlers=[handler])

# PATH FUNCTIONS:

def patch_indices(W, patch_size, overlap):
    assert overlap < patch_size*0.5, ' overlap should be at least half of patch size to be sure no repeated partches'
    if W <= patch_size:
        indices = [(0,W)]
        patch_size = W
        return indices, patch_size
    n = W // (patch_size-overlap)
    correct = W % (patch_size - overlap) - patch_size
    small = correct // n
    indices = [] # collection of tuples
    for i in range(n):
        indices.append((i*(patch_size - overlap + small), \
                       i*(patch_size - overlap + small) + patch_size))
#         indices.append((i*(patch_size - overlap + small),i*(patch_size - overlap + small) + patch_size))
    indices.append((n*(patch_size - overlap) + correct, n*(patch_size - overlap) + correct + patch_size))
    return indices, patch_size

def make_patches(image, patch_size, overlap=0, device=None):
    patch_size_y, patch_size_x = patch_size

    C, H, W = image.shape[-3], image.shape[-2], image.shape[-1]
    x_indices, patch_size_x = patch_indices(W, patch_size_x, overlap)
    y_indices, patch_size_y = patch_indices(H, patch_size_y, overlap)

    patches = torch.zeros((len(y_indices), len(x_indices), C, patch_size_y, patch_size_x), device=device)
    for i, (ay,by) in enumerate(y_indices):
        for j, (ax,bx) in enumerate(x_indices):
            patches[i,j] = image[..., ay:by, ax:bx]
    return patches, (patch_size_y, patch_size_x)
def join_patches(patches, image_shape, patch_size, overlap, device=None):
    patch_size_y, patch_size_x = patch_size

    image = torch.zeros(image_shape, device=device)
    # get indices
    H, W = image.shape[-2], image.shape[-1]
    x_indices, patch_size_x = patch_indices(W, patch_size_x, overlap)
    y_indices, patch_size_y = patch_indices(H, patch_size_y, overlap)

    for i, (ay,by) in enumerate(y_indices):
        for j, (ax,bx) in enumerate(x_indices):
            image[..., ay:by, ax:bx] =  patches[i,j] 
    return image

def change_models_to_names(model_path, to_equal_sign=True):
    if to_equal_sign:
        print('Changing from EQ to =')
    else:
        print('Changing from = to EQ')
    model_path = pathlib.Path(model_path)
    for f in model_path.glob('*.pt'):
        fname = f.name
        if to_equal_sign:
            if "EQ" in fname:
                fname = fname.replace("EQ", "=")
                f_new = pathlib.Path(f.parent.joinpath(fname))
                f.rename(f_new)  # TODO: if ever use windows! make try catch
        else:
            if "=" in fname:
                fname = fname.replace("=", "EQ")
                f_new = pathlib.Path(f.parent.joinpath(fname))
                f.rename(f_new)  # TODO: if ever use windows! make try catch


