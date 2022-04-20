from torchvision.transforms import ToPILImage, ToTensor
import torch
from random import randint

from PIL import Image


def some_random_func():
    img = ToTensor()(Image.open("test_dataset.png"))
    w = 128

    rand_x = randint(0, img.size()[1] - w * 3)
    rand_y = randint(0, img.size()[2] - w * 3)

    x_input = img[
        :,
        rand_y + w : rand_y + w * 2,
        rand_x + w : rand_x + w * 2,
    ]

    y_inputs = [
        img[
            :,
            rand_y + w * slidey : rand_y + w * (slidey + 1),
            rand_x + w * slidex : rand_x + w * (slidex + 1),
        ]
        for slidex, slidey in [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
    ]

    actual_y = torch.concat(y_inputs)
    print(actual_y.size())

    for idx, i in enumerate(y_inputs):
        t: Image = ToPILImage()(i)
        t.save(f"img_{idx}.jpg")
    ToPILImage()(x_input).save("final.jpg")
