# Image Colorizer

COS 429 Final Project

## Models

* Model 1: Sequential CNN (Simple CNN + Upsampling) with L2 loss
    - ```ImageColorizerNetwork(
        (conv_stack): Sequential(
                (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (3): ReLU()
                (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
                (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (7): ReLU()
                (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
                (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (11): ReLU()
                (12): Upsample(scale_factor=2.0, mode='nearest')
                (13): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
                (14): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (15): ReLU()
                (16): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
                (17): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (18): ReLU()
                (19): Upsample(scale_factor=2.0, mode='nearest')
                (20): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (21): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (22): ReLU()
                (23): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (24): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (25): ReLU()
                (26): Upsample(scale_factor=2.0, mode='nearest')
                (27): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (28): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (29): ReLU()
            )
        )
        ```
    - L2 loss is calculated per pixel
    - Predicted images have a heavy sepia tint. This is likely due to the fact that the L2 penalty is excessively penalizing the model. Desaturated colord tend to have a lower L2 loss than saturated colors.
    