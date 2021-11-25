import os

import numpy as np


def checkpoint(file_name, **kwargs):
    os.makedirs('checkpoints/', exist_ok=True)
    np.savez_compressed(f'checkpoints/{file_name}', **kwargs)
