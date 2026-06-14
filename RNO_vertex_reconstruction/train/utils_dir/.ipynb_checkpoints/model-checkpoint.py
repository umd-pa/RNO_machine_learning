import torch
from torch import nn

class VertexFinder1_0_0(nn.Module):
    """
    Small convolutional network that reduces a 2D input image into a 3-dimensional position vector.

    Parameters
    ----------
    input_shape : int
        Number of input channels (If only voltage, channels = 1)
    hidden_units : int
        Number of filters in the intermediate conv layers.
    output_shape : int
        Number of output channels (number of output dimensions).
    num_epochs : int
        Number of epochs this model was trained in (Unless stopped prematurely)
    batch_size : int
        Number of data points in each batch of the DataLoader
    num_train_batches : int
        Number of train batches in the DataLoader. Can be used with batch_size to figure out the total number of samples
    """
    def __init__(self,input_shape: int, hidden_units: int, output_shape: int, num_epochs: int = None, batch_size: int = None, num_train_batches: int = None):
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches
        super().__init__()
        KERNEL_HEIGHT = 2
        KERNEL_WIDTH = 33
        KERNEL_SIZE = (KERNEL_HEIGHT,KERNEL_WIDTH)

        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=1),
            nn.ReLU(),
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,8))
        )
        self.final_conv=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=KERNEL_SIZE,
                      stride=1),
            nn.AdaptiveAvgPool2d((1,1)) # reduce spatial dims to (1,1) regardless of current H/W
        )

        # ScaledTanh should accept input shaped (batch, output_shape, 1, 1)
        self.scaled_tanh = ScaledTanh(scale=3000)

    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.final_conv(x)
        x = self.scaled_tanh(x)
        x = torch.squeeze(x)
        
        return x

class ScaledTanh(nn.Module):
    """
    Custom activation function that applies tanh followed by scaling.
    
    Useful for regression tasks where you want to bound the output within a specific range. 
    In our case, our vertex coordinates should range from -3000 to +3000.
    
    Mathematical operation: f(x) = tanh(x) * scale
    Output range: [-scale, +scale]
    """
    def __init__(self, scale: float = 1.0):
        """
        Initialize the ScaledTanh activation.
        
        Args:
            scale (float): Scaling factor applied after tanh activation.
                          Output will be bounded to [-scale, +scale].
                          Should typically match the expected range of your targets.
        """
        super().__init__()
        self.scale = scale

        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply scaled tanh activation.
        
        Args:
            x (torch.Tensor): Input tensor of any shape
            
        Returns:
            torch.Tensor: Output tensor with same shape as input,
                         values bounded to [-scale, +scale]
        """
        return torch.tanh(x) * self.scale