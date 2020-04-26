import numpy as np
from enum import Enum


class PrtLvl(Enum):
    Mute     = 1
    Concise  = 2
    Detailed = 3
    Verbose  = 4


def print_level(prtl, PRT):
    if prtl.value >= PRT.value:
        return True
    else:
        return False

def throw_dice(dice):
    atry =  np.random.random_sample()
    if atry < dice:
        return True
    else:
        return False
