import torch
from torch import nn
from typing import Tuple
from numpy import ndarray

print('In models.py!')

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
            nn.LeakyReLU(0.1),
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=1),
            nn.LeakyReLU(0.1),
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


class RNO_four_1_0_0(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            # nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            # nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            # nn.BatchNorm3d(output_shape),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_0_0_tanh(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with tanh scaling.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            # nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            # nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            # nn.BatchNorm3d(output_shape),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.scaled_tanh = ScaledTanh(scale=3900) # Self-made function For now, disable

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = self.scaled_tanh(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            # nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            # nn.BatchNorm3d(hidden_units),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            # nn.BatchNorm3d(output_shape),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_2_0_0_normalized(nn.Module):
    """
    Original convolutional network using Global Pooling.
    
    Architecture:
        - Input -> [Batch Norm (Optional)]
        - Local Conv Block (Conv3d + LeakyReLU)
        - Mid Conv Block (Conv3d + LeakyReLU + MaxPool)
        - Global Conv Block (Conv3d + LeakyReLU + MaxPool)
        - Output -> Adaptive Global Pooling (Max or Avg) -> Squeeze
    
    NOTE: This architecture uses Global Pooling, which makes it translationally invariant.
    It is excellent for detecting *if* a feature exists, but poor at determining
    the exact spatial coordinate of that feature.
    """

    def __init__(self, input_shape: int,
                 hidden_units: int, 
                 output_shape: int, 
                 normalize_inputs: bool,
                 final_pool: str = 'max',
                 leak_factor: float = 0.1,
                 affine: bool = True, 
                 num_epochs: int | None = None, 
                 batch_size: int | None = None, 
                 num_train_batches: int | None = None):
        
        super().__init__()
        
        # Store training metadata for record-keeping
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches

        # --- Block 1: Local Features ---
        # Learns immediate relationships between timebins/channels.
        local_layers = []
        if normalize_inputs:
            # BatchNorm normalizes inputs to mean=0, std=1 to stabilize training
            local_layers.append(nn.BatchNorm3d(input_shape, affine=affine))
        
        local_layers.append(nn.Conv3d(in_channels=input_shape,
                                      out_channels=hidden_units,
                                      kernel_size=(2, 4, 1),
                                      padding=(0, 2, 0)))
        local_layers.append(nn.LeakyReLU(leak_factor))
        self.local_conv_block = nn.Sequential(*local_layers)
        
        # --- Block 2: Intermediate Features ---
        # Aggregates features across stations and larger time windows.
        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2, 4, 2)),
            nn.LeakyReLU(leak_factor),
            # Pooling reduces spatial size to capture wider context
            nn.MaxPool3d(kernel_size=(2, 4, 2))
        )

        # --- Block 3: Global Features ---
        # Final convolution before reducing to output vector.
        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2, 8, 1)),
            nn.LeakyReLU(leak_factor),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        # --- Final Reduction ---
        # Reduces the remaining (D, H, W) dimensions to a single (1, 1, 1) point.
        # This provides Translation Invariance.
        if final_pool.lower() == 'avg':
            self.final_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif final_pool.lower() == 'max':
            self.final_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            raise ValueError('final pool value can only be max or avg')

    def forward(self, x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        
        # Pool entire feature map to 1 point per channel
        x = self.final_pool(x)
        
        # Remove dimensions of size 1 (D, H, W) -> Result is (Batch, Channels)
        x = torch.squeeze(x)
        
        return x

class RNO_four_2_1_0_linear_fixed(nn.Module):
    """
    Modified convolutional network that replaces Global Pooling with a Dense (Linear) Layer.
    
    Purpose:
    Unlike the pooling version, this model preserves spatial information. By flattening
    the feature maps and passing them through a Linear layer, the model can learn
    specific mappings from a coordinate (x, y, z) in the input to a value in the output.
    
    Best for:
    - Regression tasks where location matters.
    - Memorization tests (overfitting checks).
    """

    def __init__(self, 
                 input_shape: int,
                 hidden_units: int, 
                 output_shape: int,
                 num_epochs: int | None = None,
                 batch_size: int | None = None,
                 num_train_batches: int | None = None,
                 input_sample_shape: Tuple[int, int, int, int] = (1, 24, 1024, 4), # REQUIRED: e.g., (1, 2, 100, 20)
                 normalize_inputs: bool = False,
                 leak_factor: float = 0.1,
                 affine: bool = True):
        """
        Args:
            input_shape: Number of input channels.
            hidden_units: Number of filters in conv layers.
            output_shape: Dimension of output vector.
            input_sample_shape: Shape of ONE input sample (C, D, H, W) used for dummy pass calculation.
            normalize_inputs: Whether to use BatchNorm.
        """

        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches

        super().__init__()
        
        # --- Block 1: Local Features ---
        local_layers = []
        if normalize_inputs:
            local_layers.append(nn.BatchNorm3d(input_shape, affine=affine))
        
        local_layers.append(nn.Conv3d(in_channels=input_shape,
                                      out_channels=hidden_units,
                                      kernel_size=(2, 4, 1),
                                      padding=(0, 2, 0)))
        local_layers.append(nn.LeakyReLU(leak_factor))
        self.local_conv_block = nn.Sequential(*local_layers)
        
        # --- Block 2: Intermediate Features ---
        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2, 4, 1)), # FIXED: Kernel_size from (2,4,2) to (2,4,1). Look at 12/23/2025 note for details.
            nn.LeakyReLU(leak_factor),
            nn.MaxPool3d(kernel_size=(2, 4, 2)) 
        )

        # --- Block 3: Global Features ---
        # Note: We keep out_channels as 'hidden_units' here to maintain capacity before flattening.
        # In the previous model, we reduced to 'output_shape' here, but now the Linear layer handles that.
        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units, 
                      kernel_size=(2, 8, 1)),
            nn.LeakyReLU(leak_factor),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        # --- DYNAMIC LINEAR LAYER CALCULATION ---
        # We perform a "dummy pass" to find out exactly how many neurons exist
        # after the convolution blocks. This avoids manual math errors.
        
        # Create dummy input with Batch Size = 1
        # input_sample_shape should be (Channels, Depth, Height, Width)
        dummy_input = torch.zeros(1, *input_sample_shape)
        
        print(f"Calculating linear layer size using dummy shape: {dummy_input.shape}")
        
        with torch.no_grad():
            # Run the dummy through the conv blocks
            d = self.local_conv_block(dummy_input)
            d = self.mid_conv_block(d)
            d = self.global_conv_block(d)
            
            # Calculate flattened size (Batch * Features) -> We want just Features
            self.flatten_size = d.view(1, -1).size(1)
            
        print(f"Flattened feature size: {self.flatten_size}")
        print(f"Creating Linear Layer: {self.flatten_size} -> {output_shape}")

        # --- Final Linear Layer ---
        # This layer maps every single remaining spatial pixel to the output coordinates.
        self.final_linear = nn.Linear(in_features=self.flatten_size, 
                                      out_features=output_shape)

    def forward(self, x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        
        # Flatten the output for the Linear layer
        # (Batch, Channels, D, H, W) -> (Batch, Flattened_Features)
        x = x.view(x.size(0), -1) 
        
        # Pass through Linear layer to get exact coordinates
        x = self.final_linear(x)
        
        return x

class RNO_four_late_linear_merge(nn.Module):
    """
    Latest model that utilizes a new archietcture for vertex reconstruction:

    Each station voltage is processed separately. For each station we use previously implemented CNN techniques and
    at the end we merge all of the input data with a single linear layer. Additionally, this model allows for variable
    time bins and dimensions.

    Station Block: Extract features of every single station.
    Normalization Block: Normalize the dimensions for managable input of the last layer.
    Linear TDoA: Utilize the smaller output of the normalization block with the extracted features to discern the actual location. 
    """

    def __init__(self, 
                 input_shape: int,
                 hidden_units: int, 
                 output_shape: int, 
                 num_epochs: int | None = None,
                 batch_size: int | None = None,
                 num_train_batches: int | None = None,
                 station_num: int = 4,
                 leak_factor: float = 0.1,
                 dropout_rate: float = 0.1):
        
        super().__init__()
        
        # Store metadata (optional but good for logging)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train_batches = num_train_batches

        # --- STATION BLOCK (Feature Extraction) ---
        self.station_block = nn.Sequential(
            # 1. Normalize Input (Raw Voltage)
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True), 

            # 2. Layer 1: Detect Pulses
            nn.Conv3d(in_channels=input_shape, out_channels=hidden_units, 
                      kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),

            # 3. Layer 2: Refine Features
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units, 
                      kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.MaxPool3d(kernel_size=(1,2,1)), # Shrink time
            
            # 4. Layer 3: Deep Features
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units, 
                      kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.MaxPool3d(kernel_size=(1,2,1))  # Shrink time
        )

        # --- NORMALIZATION BLOCK ---
        # Forces output to (24 Depth, 128 Time, n Stations)
        self.normalization_block = nn.Sequential(
            nn.AdaptiveMaxPool3d((24, 128, station_num))
        )

        # --- LINEAR TDOA BLOCK ---
        # Channels (hidden) * Depth (24) * Time (128) * Stations (station_num)
        self.flatten_size = hidden_units * 24 * 128 * station_num
        
        self.linear_TDoA = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=self.flatten_size, out_features=output_shape)
        )

    def forward(self, x):
        x = self.station_block(x)
        x = self.normalization_block(x)
        x = self.linear_TDoA(x)
        x = torch.squeeze(x)
        return x


class RNO_four_late_non_linear_merge(nn.Module):
    """
    Latest model that utilizes a new archietcture for vertex reconstruction:

    Each station voltage is processed separately. For each station we use previously implemented CNN techniques and
    at the end we merge all of the input data with a single linear layer. Additionally, this model allows for variable
    time bins and dimensions.

    Station Block: Extract features of every single station.
    Normalization Block: Normalize the dimensions for managable input of the last layer.
    NON-Linear TDoA: Utilize the smaller output of the normalization block with the extracted features to discern the actual location, with added non-linearity
    """

    def __init__(self, 
                 input_shape: int,
                 hidden_units: int, 
                 output_shape: int,
                 label_mean: ndarray,
                 label_std: ndarray,
                 num_epochs: int | None = None,
                 batch_size: int | None = None,
                 num_train_batches: int | None = None,
                 station_num: int = 4,
                 leak_factor: float = 0.1,
                 dropout_rate: float = 0.1,
                 temporal_res: int = 64):
        
        super().__init__()
        
        # Store metadata (optional but good for logging)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train_batches = num_train_batches

        # Store statistics
        if label_mean is None:
            label_mean = torch.zeros(output_shape)
        if label_std is None:
            label_std = torch.ones(output_shape)
            
        # Register them into the model's state
        self.register_buffer('label_mean', torch.tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std', torch.tensor(label_std, dtype=torch.float32))

        # --- STATION BLOCK (Feature Extraction) ---
        self.station_block = nn.Sequential(
            # 1. Normalize Input (Raw Voltage)
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True), 

            # 2. Layer 1: Detect Pulses
            nn.Conv3d(in_channels=input_shape, out_channels=hidden_units, 
                      kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),

            # 3. Layer 2: Refine Features
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units, 
                      kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.MaxPool3d(kernel_size=(1,2,1)), # Shrink time
            
            # 4. Layer 3: Deep Features
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units, 
                      kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),
        )

        # --- NORMALIZATION BLOCK ---
        # Forces output to (24 Depth, 128 Time, n Stations)
        self.normalization_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units, out_channels=1, kernel_size=1), # Kill hidden units!
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.AdaptiveMaxPool3d((24, temporal_res, station_num))
        )

        # --- LINEAR TDOA BLOCK ---
        # Channels (hidden) * Depth (24) * Time (temporal_res) * Stations (station_num)
        self.flatten_size = 1 * 24 * temporal_res * station_num
        
        self.nonlinear_TDoA = nn.Sequential(
            nn.Flatten(),

            # Layer 1: Feature compression
            nn.Linear(in_features=self.flatten_size, out_features=256),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),
            
            # Layer 2: Geometric Reasoning (Triangulation)
            # This layer allows the model to perform the non-linear math
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Output
            nn.Linear(in_features=128, out_features=output_shape)
        )

    def forward(self, x, return_unnormalized=False):
            """
            Forward pass of the model.
            
            Args:
                x (torch.Tensor): Input voltages/images.
                return_unnormalized (bool): If True, the model will 
                                        automatically undo its own normalization math 
                                        and output raw physical coordinates (meters).
            """
            # Pass data through your network blocks
            x = self.station_block(x)
            x = self.normalization_block(x)
            x = self.nonlinear_TDoA(x)
            
            # Ensure the output is shape (Batch, 3)
            x = torch.squeeze(x)
            
            # If we are in final evaluation mode, convert the network's 
            # [-1, 1] normalized output back into real-world physical units.
            if return_unnormalized:
                # Broadcasts the math perfectly across the batch dimension
                x = (x * self.label_std) + self.label_mean
                
            return x


class RNO_four_1_1_1s_batch_norm_dropout_extraconv_nonleak(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

    def __init__(self,input_shape: int, hidden_units: int,  output_shape: int, affine: bool = True, num_epochs: int = None, batch_size: int = None, num_train_batches: int = None):
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches
        super().__init__()

        self.local_conv_block = nn.Sequential(
            nn.BatchNorm3d(input_shape,momentum=0.01,affine=affine),
            torch.nn.Dropout3d(p=0.1),

            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),

            nn.BatchNorm3d(hidden_units,momentum=0.01,affine=affine),
            torch.nn.Dropout3d(p=0.1),
            nn.ReLU(),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            nn.BatchNorm3d(hidden_units,momentum=0.01,affine=affine),
            torch.nn.Dropout3d(p=0.1),
            nn.ReLU(),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            nn.BatchNorm3d(hidden_units,momentum=0.01,affine=affine),
            nn.ReLU(),
        )

        self.mid_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),

            nn.BatchNorm3d(hidden_units,momentum=0.01,affine=affine),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,4,2)),
            nn.BatchNorm3d(hidden_units,momentum=0.01,affine=affine)
        )

        self.global_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),

            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_dropout_extraconv_leaky(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

    def __init__(self,input_shape: int, hidden_units: int,  output_shape: int, affine: bool = True, num_epochs: int = None, batch_size: int = None, num_train_batches: int = None):
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches
        super().__init__()

        self.local_conv_block = nn.Sequential(
            torch.nn.Dropout3d(p=0.1),

            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),

            torch.nn.Dropout3d(p=0.1),
            nn.LeakyReLU(0.1),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            torch.nn.Dropout3d(p=0.1),
            nn.LeakyReLU(0.1),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            nn.LeakyReLU(0.1),
        )

        self.mid_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),

            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),

            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_dropout_extraconv_noleak(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

    def __init__(self,input_shape: int, hidden_units: int,  output_shape: int, affine: bool = True, num_epochs: int = None, batch_size: int = None, num_train_batches: int = None):
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches
        super().__init__()

        self.local_conv_block = nn.Sequential(
            torch.nn.Dropout3d(p=0.1),

            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),

            torch.nn.Dropout3d(p=0.1),
            nn.ReLU(),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            torch.nn.Dropout3d(p=0.1),
            nn.ReLU(),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            nn.ReLU(),
        )

        self.mid_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),

            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),

            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_dropout_leaky(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

    def __init__(self,input_shape: int, hidden_units: int,  output_shape: int, affine: bool = True, num_epochs: int = None, batch_size: int = None, num_train_batches: int = None):
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches
        super().__init__()

        self.local_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),

            torch.nn.Dropout3d(p=0.1),
            nn.LeakyReLU(0.1),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            nn.LeakyReLU(0.1),

        )

        self.mid_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),

            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),

            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        x = self.local_conv_block(x)
        # print(f'After local block: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}')
        # print('------------------------------------------------------------------------------------')
        
        x = self.mid_conv_block(x)
        # print(f'After mid block: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}')
        # print('------------------------------------------------------------------------------------')
        
        x = self.global_conv_block(x)
        # print(f'After global block: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}')
        # print('------------------------------------------------------------------------------------')
        
        x = self.final_pool(x)
        # print(f'After final pool: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}')
        
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_dropout_noleak(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

    def __init__(self,input_shape: int, hidden_units: int,  output_shape: int, affine: bool = True, num_epochs: int = None, batch_size: int = None, num_train_batches: int = None):
        self.hidden_units = hidden_units
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if num_train_batches is not None:
            self.num_train_batches = num_train_batches
        super().__init__()

        self.local_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),

            torch.nn.Dropout3d(p=0.1),
            nn.ReLU(),

            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),

            nn.ReLU(),

        )

        self.mid_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),

            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(

            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),

            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_extraconv(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_extraconv_dropout(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1)
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_extraconv_dropout_211(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.2),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1)
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        
        return x


class RNO_four_1_1_1s_extra_extraconv_dropout_2211(nn.Module):
    """
    Improved convolutional network that reduces a 3D input image into a 3-dimensional position vector with batchnorm.
    Learning is separated in steps:
    Local layer: Learns voltage shape between timebins + channels
    Mid layer: Learns voltage shape between timebins + channels + stations
    Global layer: Learns voltage position in time relative to the overall time array.

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

        self.local_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 2, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.2),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.2),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=(2, 4, 1),
                    padding=(0, 1, 0)),
            nn.LeakyReLU(0.1),
            torch.nn.Dropout3d(p=0.1)
        )

        self.mid_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=(2,4,2)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2,4,2))
        )

        self.global_conv_block = nn.Sequential(
            nn.Conv3d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=(2,8,1)),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 8, 1))
        )
        
        self.final_pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x):
        x = self.local_conv_block(x)
        x = self.mid_conv_block(x)
        x = self.global_conv_block(x)
        x = self.final_pool(x)
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
    
print('finished reading model.py!')