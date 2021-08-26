import torchvision.transforms
from PIL import Image
import numpy as np
import torch
import json
import torchvision.transforms.functional as F
from torch import Tensor

torch.manual_seed(1337)
np.random.seed(1337)


def circle(r, nrows=32, ncols=32):
    base = np.ones((nrows, ncols))
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = 32 / 2, 32 / 2
    disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (r / 2)**2)
    base[disk_mask] = 0
    return base


def rect(width, height, nrows=32, ncols=32):
    base = np.zeros((nrows, ncols))
    y_cnt = nrows//2
    x_cnt = ncols//2
    base[y_cnt-height//2:y_cnt-height//2+height,
         x_cnt-width//2:x_cnt-width//2+width] = 1
    return base


def cross(x_width, x_height, y_width, y_height, nrows=32, ncols=32):
    base = np.zeros((nrows, ncols))
    y_cnt = nrows//2
    x_cnt = ncols//2
    base[y_cnt-x_width//2:y_cnt-x_width//2+x_width,
         x_cnt-x_height//2:x_cnt-x_height//2+x_height] = 1
    base[y_cnt-y_height//2:y_cnt-y_height//2+y_height,
         x_cnt-y_width//2:x_cnt-y_width//2+y_width] = 1
    return base


def generate_random_params(num):
    circle = np.random.randint(5, 20, (num,), dtype=np.int8)
    circle_thickness = np.random.randint(1, 12, (num,), dtype=np.int8)
    rect = np.random.randint(5, 20, (num, 2,), dtype=np.int8)
    rect_thickness = np.random.randint(1, 12, (num,), dtype=np.int8)
    cross_h = np.random.randint(15, 20, (num, 2,), dtype=np.int8)
    cross_w = np.random.randint(5, 10, (num, 2,), dtype=np.int8)
    cross = np.zeros((num, 4), dtype=np.int8)
    cross[:, 0] = cross_w[:, 0]
    cross[:, 1] = cross_h[:, 0]
    cross[:, 2] = cross_w[:, 1]
    cross[:, 3] = cross_h[:, 1]
    cross_thickness = np.random.randint(1, 12, (num,), dtype=np.int8)
    return (circle, circle_thickness), (rect, rect_thickness), (cross, cross_thickness)


class RandomAffine(torchvision.transforms.RandomAffine):

    def __init__(self):
        super().__init__(
            degrees=45,
            translate=(0.1, 0.1),
            shear=40,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )

    def forward(self, img):

        img = torch.tensor(img).unsqueeze(0)

        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate,
                              self.scale, self.shear, img_size)

        affined = F.affine(
            img, *ret,
            interpolation=self.interpolation,
            fill=fill
        )
        affined[affined < 0.5] = 0
        affined[affined >= 0.5] = 1

        return affined.numpy()[0], ret


if __name__ == "__main__":

    affine = RandomAffine()
    geometrys = generate_random_params(3000)
    factorys = [
        circle, rect, cross
    ]
    shapes = [
        "circle", "rectangle", "cross"
    ]
    i = 0
    geometry_params = []
    for (factory, params, shape) in zip(factorys, geometrys, shapes):
        for (param, thickness) in zip(*params):
            if type(param) == np.int8:
                img = factory(param)
            else:
                img = factory(*param)
            img_affine, ret = affine(img)
            filename = "{}_{}.png".format(i, thickness)
            Image.fromarray(
                img_affine*255
            ).convert('L').save("img/{}".format(filename))

            if shape == "circle":
                geometry_param = {
                    "r": int(param)
                }
            elif shape == "rectangle":
                geometry_param = {
                    "width": int(param[0]),
                    "height": int(param[1])
                }
            else:
                geometry_param = {
                    "x_width": int(param[0]),
                    "x_height": int(param[1]),
                    "y_width": int(param[2]),
                    "y_height": int(param[3]),
                }
            geometry_param.update({
                "angle": ret[0],
                "x_translation": ret[1][0],
                "y_translation": ret[1][1],
                "shear": ret[3][0]
            })
            geometry_param.update({
                "shape": shape,
                "thickness": int(thickness),
                "filename": filename
            })
            geometry_params.append(geometry_param)

            i += 1

    with open("img/geometry_params.json", "w") as fp:
        json.dump(geometry_params, fp)
