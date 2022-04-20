import os
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import torch.optim as optim

from unet import UNet

from random import seed, randint

from datetime import datetime

from PIL import Image

import typing

import json


class DreamDataset(Dataset):
    def __init__(
        self, image_list: typing.List[str], sample_count: int, size: int = 200
    ):

        seed(int(datetime.now().timestamp()))
        self.sample_count = sample_count

        self.image_to_tensor = ToTensor()

        for f in image_list:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"File {f} does not exist !")

        # Get a list of tensors out of the list of images
        self.image_list = [
            self.image_to_tensor(Image.open(filename)) for filename in image_list
        ]

        self.input_shape = size

    def __len__(self):
        return len(self.image_list) * self.sample_count

    def __getitem__(self, idx):
        img_idx = idx // self.sample_count
        img = self.image_list[img_idx]

        # Pick a random square on the chosen image, with side input_shape
        rand_x = randint(0, img.size(2) - self.input_shape * 3)
        rand_y = randint(0, img.size(1) - self.input_shape * 3)

        input_x = img[
            :,
            rand_y + self.input_shape : rand_y + self.input_shape * 2,
            rand_x + self.input_shape : rand_x + self.input_shape * 2,
        ]

        gt_y = [
            img[
                :,
                rand_y
                + self.input_shape * slidey : rand_y
                + self.input_shape * (slidey + 1),
                rand_x
                + self.input_shape * slidex : rand_x
                + self.input_shape * (slidex + 1),
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

        return input_x, torch.concat(gt_y)


optimizers = {
    "SGD": torch.optim.SGD,
    "Adagrad": torch.optim.Adagrad,
    "Adamax": torch.optim.Adamax,
}

loss_functions = {"L1Loss": nn.L1Loss, "HuberLoss": nn.HuberLoss}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_args: dict,
    epoch: int,
    loss_name: str,
    checkpoint_path: str,
):
    # Always serialize/deserialize state dicts to cpu
    device = torch.device("cpu")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_name": optimizer.__class__.__name__,
            "optimizer_args": optimizer_args,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_name": loss_name,
        },
        checkpoint_path,
    )


def load_from_checkpoint(checkpoint_path: str):
    device = torch.device("cpu")
    with open(checkpoint_path, "rb") as f:
        return torch.load(f, map_location=device)


def init_training_from_scratch(args, device: torch.device, net: torch.nn.Module):

    loss_name = args.loss_function

    criterion: nn.Module = loss_functions[loss_name]()
    criterion.to(device)

    net.to(device)

    optimizer_args: dict = args.optimizer_args or {"lr": 0.001}
    optimizer = optimizers[args.optimizer](net.parameters(), **optimizer_args)

    starting_epoch = 0

    return optimizer, starting_epoch, net, criterion, optimizer_args, loss_name


def init_training_from_checkpoint(args, device: torch.device, net: torch.nn.Module):

    checkpoint_data = load_from_checkpoint(args.checkpoint)

    loss_name = checkpoint_data["loss_name"]
    criterion: nn.Module = loss_functions[loss_name]()
    criterion.to(device)

    net.to(device)

    optimizer_args: dict = args.optimizer_args or checkpoint_data["optimizer_args"]
    optimizer: torch.optim.Optimizer = optimizers[checkpoint_data["optimizer_name"]](
        net.parameters(), **optimizer_args
    )
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    net.load_state_dict(checkpoint_data["model_state_dict"])
    net.train()

    starting_epoch = checkpoint_data["epoch"] + 1

    return optimizer, starting_epoch, net, criterion, optimizer_args, loss_name


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNet(3, 24)

    dataset = DreamDataset(
        [r"C:\Users\Charles\Dev\MachineLearning\ImageCompleter\test_dataset.png"],
        256,
        128,
    )
    trainloader = DataLoader(dataset, batch_size=32)

    save_frequency = args.save_frequency

    if args.checkpoint:
        (
            optimizer,
            starting_epoch,
            net,
            criterion,
            optimizer_args,
            loss_name,
        ) = init_training_from_checkpoint(args, device, net)
    else:
        (
            optimizer,
            starting_epoch,
            net,
            criterion,
            optimizer_args,
            loss_name,
        ) = init_training_from_scratch(args, device, net)

    with open(f"{args.save_path}/loss_evolution.csv", "a") as file_log:
        for idx, epoch in enumerate(
            range(starting_epoch, starting_epoch + args.epochs), 1
        ):
            running_loss = 0.0
            for inx_batch, gt_batch in trainloader:

                inx_batch = inx_batch.to(device)
                gt_batch = gt_batch.to(device)

                optimizer.zero_grad()

                outputs = net(inx_batch)
                loss = criterion(outputs, gt_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if idx % save_frequency == 0:
                save_checkpoint(
                    net,
                    optimizer,
                    optimizer_args,
                    epoch,
                    loss_name,
                    f"{args.save_path}/checkpoint_epoch_{epoch}.tar",
                )

            print(f"Epoch {epoch} loss: {running_loss}")
            file_log.write(f"{epoch},{running_loss}\n")

        save_checkpoint(
            net,
            optimizer,
            optimizer_args,
            epoch,
            loss_name,
            f"{args.save_path}/checkpoint_epoch_{epoch}.tar",
        )

        print("Finished Training")
        return 0


def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = None

    if args.checkpoint:
        net = UNet(3, 24)
        with open(args.checkpoint, "rb") as f:
            checkpoint_data = torch.load(f, map_location=device)
            net.load_state_dict(checkpoint_data["model_state_dict"])
            net.eval()

    elif args.model:
        with open(args.model, "rb") as f:
            net: nn.Module = torch.load(f, map_location=device)
            assert isinstance(
                net, nn.Module
            ), f"Fatal: {args.model} does not serialize a pytorch module !"
            net.eval()
    else:
        print("No module provided for evaluation, exiting.")
        return 1

    if net is None:
        print("Something went wrong while loading the module, no module loaded !")
        return 1

    img = ToTensor()(Image.open(args.image)).unsqueeze(0)[:, :, 200:400, 100:300]
    img.to(device)

    with torch.no_grad():
        result: torch.Tensor = net(img)[0]
        result_NW = result[0:3, :, :]
        result_W = result[3:6, :, :]
        result_SW = result[6:9, :, :]
        result_N = result[9:12, :, :]
        result_S = result[12:15, :, :]
        result_NE = result[15:18, :, :]
        result_E = result[18:21, :, :]
        result_SE = result[21:24, :, :]

        stripe1 = torch.concat([result_NW, result_N, result_NE], dim=2)
        stripe2 = torch.concat([result_W, img[0], result_E], dim=2)
        stripe3 = torch.concat([result_SW, result_S, result_SE], dim=2)

        final_render = torch.concat([stripe1, stripe2, stripe3], dim=1)

        final_image: Image = ToPILImage()(final_render)
        final_image.save("final_render.tif")

    return 0


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "epochs", help="How many epochs to train the model for", type=int
    )
    train_parser.add_argument(
        "--checkpoint", default=None, help="Path to a checkpoint file"
    )

    train_parser.add_argument(
        "--optimizer",
        help="What optimizer to use for training",
        default="Adamax",
        choices=optimizers.keys(),
    )
    train_parser.add_argument(
        "--optimizer-options",
        help="Optimizer parameters such as learning rate, momentum, etc",
        type=json.loads,
        dest="optimizer_args",
        default=None,
    )
    train_parser.add_argument("--save-path", default="saves")
    train_parser.add_argument(
        "--loss-function", default="L1Loss", choices=loss_functions.keys()
    )
    train_parser.add_argument("--save-frequency", default=100)

    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument(
        "image", help="Path to an image you'd like to autocomplete"
    )
    eval_parser.add_argument("--checkpoint", help="Path to a checkpoint file")
    eval_parser.add_argument("--model", help="Path to a final module file")

    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()
    exit(args.func(args))
