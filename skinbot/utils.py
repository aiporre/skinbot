import logging
import os
import torch


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
        logging.basicConfig(level=numeric_level)

# PATH FUNCTIONS:

def patch_indices(W, patch_size, overlap):
    n = W // (patch_size-overlap)
    correct = W % (patch_size - overlap) - patch_size
    small = correct // n
    indices = [] # collection of tuples
    for i in range(n):
        indices.append((i*(patch_size - overlap + small), \
                       i*(patch_size - overlap + small) + patch_size))
#         indices.append((i*(patch_size - overlap + small),i*(patch_size - overlap + small) + patch_size))
    indices.append((n*(patch_size - overlap) + correct, n*(patch_size - overlap) + correct + patch_size))
    return indices

def make_patches(image, patch_size, overlap=0):
    C, H, W = image.shape[-3], image.shape[-2], image.shape[-1]
    x_indices = patch_indices(W, patch_size, overlap)
    y_indices = patch_indices(H, patch_size, overlap)
    patches = torch.zeros((len(y_indices), len(x_indices), C, patch_size, patch_size))
    for i, (ay,by) in enumerate(y_indices):
        for j, (ax,bx) in enumerate(x_indices):
            patches[i,j] = image[..., ay:by, ax:bx]
    return patches
def join_patches(patches, image_shape, patch_size, overlap):
    image = torch.zeros(image_shape)
    # get indices
    H, W = image.shape[-2], image.shape[-1]
    x_indices = patch_indices(W, patch_size, overlap)
    y_indices = patch_indices(H, patch_size, overlap)
    for i, (ay,by) in enumerate(y_indices):
        for j, (ax,bx) in enumerate(x_indices):
            image[..., ay:by, ax:bx] =  patches[i,j] 
    return image


