''' some tensor images transformation
'''
import torch
from PIL import Image

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2)
), 1)

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)

pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), Image.BOX).resize((32, 32), Image.BOX)

rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)

invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, :], 1 - x[2:, :, :]), 0)


def get_shift_transform(name):
    return {
        'jigsaw': jigsaw,
        'speckle': speckle,
        'pixelate': pixelate,
        'rgb_shift': rgb_shift,
        'invert': invert
    }[name]