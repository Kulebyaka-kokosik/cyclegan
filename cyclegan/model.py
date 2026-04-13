import torch
from torch import nn

class DownScaler(nn.Module):
    def __init__(self, in_channels: int):
        super(DownScaler, self).__init__()
        seq = [
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(2 * in_channels),
            nn.ReLU(inplace=True),
        ]
        self._down_scaler = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._down_scaler(x)


class DownScalersBlock(nn.Module):

    def __init__(self, in_channels, n_twice: int):
            super(DownScalersBlock, self).__init__()
            self._down_scalers = nn.Sequential(*[DownScaler(2 ** i * in_channels) for i in range(n_twice)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._down_scalers(x)


class UpScaler(nn.Module):
    def __init__(self, in_channels: int):
        super(UpScaler, self).__init__()
        seq = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        self._up_scaler = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._up_scaler(x)


class UpScalersBlock(nn.Module):
    def __init__(self, in_channels: int, n_twice: int):
            super(UpScalersBlock, self).__init__()
            self._up_scalers = nn.Sequential(*[UpScaler(in_channels=in_channels // (2 ** i)) for i in range(n_twice)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._up_scalers(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int):
        super(ResidualBlock, self).__init__()
        seq = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size),
            nn.InstanceNorm2d(channels),
        ]
        self._res_block = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._res_block(x)

class ResidualBlocks(nn.Module):
    def __init__(self, channels: int, n_blocks: int):
        super(ResidualBlocks, self).__init__()
        seq = []
        seq.extend([ResidualBlock(channels=channels, kernel_size=3) for _ in range(n_blocks)])
        self._res_blocks = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._res_blocks(x)


class Generator(nn.Module):
    def __init__(
            self,
            n_twice: int = 2,
            in_channels: int = 3,
            work_channels: int = 64,
            blocks_number: int = 9,
    ):
        super(Generator, self).__init__()
        self._connection_to_work_channels = nn.Sequential(
            *[
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=in_channels, out_channels=work_channels, kernel_size=7),
                nn.InstanceNorm2d(work_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self._down_scaler_block = DownScalersBlock(n_twice=n_twice, in_channels=work_channels)
        self._res_blocks = ResidualBlocks(channels=work_channels * 2 ** n_twice, n_blocks=blocks_number)
        self._up_scaler_block = UpScalersBlock(n_twice=n_twice, in_channels=work_channels * 2 ** n_twice)
        self._connection_to_in_channels = nn.Sequential(
            *[
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=work_channels, out_channels=in_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._connection_to_work_channels(x)
        x = self._down_scaler_block(x)
        x = self._res_blocks(x)
        x = self._up_scaler_block(x)
        x = self._connection_to_in_channels(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, work_channels: int = 64):
        super(Discriminator, self).__init__()
        seq = list()
        seq.append(nn.Conv2d(in_channels=in_channels, out_channels=work_channels, kernel_size=4, padding=1, stride=2))
        seq.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(2):
            seq.extend(
                [
                    nn.Conv2d(in_channels=work_channels * 2 ** i, out_channels=work_channels * 2 ** (i + 1), kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(work_channels * 2 ** (i + 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        seq.extend(
            [
                nn.Conv2d(in_channels=work_channels * 2 ** 2, out_channels=work_channels * 2 ** 3, kernel_size=4, padding=1),
                nn.InstanceNorm2d(work_channels * 2 ** 3),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        seq.append(nn.Conv2d(in_channels=work_channels * 2 ** 3, out_channels=1, kernel_size=4, padding=1))
        self._discriminator = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._discriminator(x)


class CycleGAN(nn.Module):
    def __init__(
            self,
            n_twice: int = 2,
            in_channels: int = 3,
            work_channels: int = 64,
            blocks_number: int = 9,
            training: bool = False,
    ):
        super(CycleGAN, self).__init__()
        self.training = training
        self.generator_A = Generator(n_twice=n_twice, in_channels=in_channels, work_channels=work_channels,
                                     blocks_number=blocks_number)
        self.generator_B = Generator(n_twice=n_twice, in_channels=in_channels, work_channels=work_channels,
                                     blocks_number=blocks_number)

        if self.training:
            self.discriminator_A = Discriminator(in_channels=in_channels, work_channels=work_channels)
            self.discriminator_B = Discriminator(in_channels=in_channels, work_channels=work_channels)
