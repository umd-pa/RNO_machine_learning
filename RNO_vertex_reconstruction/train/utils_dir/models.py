from .my_utils import auto_name
import torch
from torch import dropout, nn
from numpy import ndarray


class RNO_four_gentle_non_linear_merge(nn.Module):
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
                 label_mean: ndarray | None | torch.Tensor = None,
                 label_std: ndarray | None | torch.Tensor = None,
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
        self.register_buffer('label_mean', torch.as_tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std', torch.as_tensor(label_std, dtype=torch.float32))
        
        # --- STATION BLOCK (Feature Extraction) ---
        self.station_block = nn.Sequential(
            # 1. Normalize raw voltages per-channel before any convolution
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True),

            # 2. Layer 1: Detect pulses at FULL time resolution (1024 bins = 8192 ns)
            # Wide time kernel to capture waveform shape before any downsampling
            nn.Conv3d(in_channels=input_shape, out_channels=hidden_units,
                    kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),
            # Pool 4x in time ONLY — 1024 → 256 bins (32 ns resolution preserved)
            # Physically justified: max meaningful TDoA ~50 bins, 4x pool keeps ~12 bins of precision
            # Reduces activation size ~4x: 65GB → ~16GB
            # Station and antenna dimensions untouched — spatial locality preserved
            nn.MaxPool3d(kernel_size=(1,4,1)),

            # 3. Layer 2: Refine features at 256 time bins — sufficient TDoA resolution
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units,
                    kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),

            # 4. Layer 3: Deep abstract features — still at 256 time bins
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units,
                    kernel_size=(3,3,1), padding=(1,1,0)),
            nn.LeakyReLU(negative_slope=leak_factor),
        )

        # --- NORMALIZATION BLOCK ---
        # Compresses hidden_units channels down to 1 for the TDoA block.
        # Two-step compression instead of one aggressive step — gentler gradient flow.
        # 64 → 16 → 1 rather than 64 → 1 in a single 1x1 conv.
        self.normalization_block = nn.Sequential(
            # Step 1: compress to hidden_units//4 channels (e.g. 64 → 16)
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units//4, kernel_size=1),
            nn.LeakyReLU(negative_slope=leak_factor),
            # Step 2: compress to 1 channel (e.g. 16 → 1)
            nn.Conv3d(in_channels=hidden_units//4, out_channels=1, kernel_size=1),
            nn.LeakyReLU(negative_slope=leak_factor),
            # Final spatial projection to fixed size for the linear TDoA block
            nn.AdaptiveMaxPool3d((24, temporal_res, station_num))
        )

        # --- LINEAR TDOA BLOCK ---
        # Input: 1 channel × 24 antennas × temporal_res bins × station_num stations
        self.flatten_size = 1 * 24 * temporal_res * station_num

        self.nonlinear_TDoA = nn.Sequential(
            nn.Flatten(),

            # Layer 1: Cross-station feature compression
            nn.Linear(in_features=self.flatten_size, out_features=256),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Layer 2: Geometric reasoning — learns TDoA triangulation relationships
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Output: normalized (x, y, z) vertex coordinates
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
                x = (x * self.label_std) + self.label_mean #type: ignore
                
            return x

class RNO_four_mixing(nn.Module):
    """
    Vertex reconstruction model with shared-weight per-station encoding and explicit cross-station fusion.

    Architecture:
        Station Encoder (shared weights): A single 2D CNN processes each of the 4 stations
                                          independently but with identical weights. This forces
                                          the model to learn "what a pulse looks like" once,
                                          rather than learning 4 separate detectors.

        TDoA Fusion Block: Concatenates the 4 per-station feature vectors in a fixed order,
                           preserving station identity. The linear layers then learn the
                           cross-station timing geometry (triangulation).

    Input shape:  (batch, 1, 24, 1024, 4)  →  1 voltage channel, 24 antennas, 1024 time bins, 4 stations
    Output shape: (batch, 3)               →  normalized (x, y, z) vertex coordinates
    """

    def __init__(self,
                input_shape: int,
                hidden_units: int,
                output_shape: int,
                label_mean=None,
                label_std=None,
                station_num: int = 4,
                leak_factor: float = 0.1,
                dropout_rate: float = 0.1):

        super().__init__()

        self.station_num = station_num  # used in forward() to loop over stations

        # Normalization statistics — registered as buffers so they are:
        #   - saved/loaded with the model's state_dict
        #   - automatically moved to the correct device with .to(device)
        #   - excluded from gradient computation
        if label_mean is None:
            label_mean = torch.zeros(output_shape)
        if label_std is None:
            label_std = torch.ones(output_shape)

        self.register_buffer('label_mean', torch.as_tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std',  torch.as_tensor(label_std,  dtype=torch.float32))

        # ====================================================================
        # STATION ENCODER (shared weights across all stations)
        # Input per station: (batch, 1, 24, 1024)
        #   dim 1 = voltage channel (1)
        #   dim 2 = antennas (24)
        #   dim 3 = time bins (1024)
        #
        # Using Conv2d instead of Conv3d because each station is a 2D signal
        # (antennas × time). Shared weights mean station 0 and station 2 are
        # processed by the exact same filters — physically correct since all
        # stations are identical hardware.
        # ====================================================================
        self.station_encoder = nn.Sequential(

            # Normalize raw voltages per station before any convolution
            nn.BatchNorm2d(input_shape, momentum=0.01, affine=True),

            # Layer 1: Detect pulses — wide time kernel to capture waveform shape
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=(3, 7), padding=(1, 3)),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.MaxPool2d(kernel_size=(1, 4)),   # 1024 → 256 time bins
            

            # Layer 2: Cross-antenna features — relationships between antenna rows
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=(3, 7), padding=(1, 3)),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.MaxPool2d(kernel_size=(2, 4)),   # 24→12 antennas, 256→64 time bins

            # Layer 3: Deep features
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=(3, 5), padding=(1, 2)),
            nn.LeakyReLU(negative_slope=leak_factor),

            # Pool to a fixed spatial size regardless of input dimensions —
            # makes the model robust to changes in temporal_res or antenna count
            nn.AdaptiveAvgPool2d((8, 32)),      # → (batch, hidden_units, 8, 32)

            nn.Flatten()                         # → (batch, hidden_units * 8 * 32)
        )

        # Size of one station's feature vector after encoding
        per_station_features = hidden_units * 8 * 32

        # ====================================================================
        # TDOA FUSION BLOCK
        # Receives 4 station feature vectors concatenated in fixed order:
        #   [station_0_features | station_1_features | station_2_features | station_3_features]
        #
        # Crucially, station identity is preserved by position in the vector.
        # The linear layers can therefore learn "signal arrived at station 0
        # before station 2" — the actual geometric triangulation signal.
        # ====================================================================
        self.nonlinear_TDoA = nn.Sequential(

            # Layer 1: Cross-station feature compression
            nn.Linear(in_features=per_station_features * station_num, out_features=256),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Layer 2: Geometric reasoning (triangulation)
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Output: normalized (x, y, z)
            nn.Linear(in_features=128, out_features=output_shape)
        )

    def forward(self, x, return_unnormalized=False):
        """
        Forward pass.

        Args:
            x (Tensor):                Input voltages, shape (batch, 1, 24, 1024, 4).
            return_unnormalized (bool): If True, converts the network's normalized output
                                        back into real-world physical coordinates (meters).
        Returns:
            Tensor: shape (batch, 3)
        """
        # Process each station independently with shared encoder weights.
        # x[..., s] slices out one station → (batch, 1, 24, 1024)
        station_features = [
            self.station_encoder(x[..., s])
            for s in range(self.station_num)
        ]

        # Concatenate in fixed station order → (batch, per_station_features * station_num)
        # Order is preserved so the TDoA block always sees station 0 at the same position
        combined = torch.cat(station_features, dim=1)

        # Cross-station geometric fusion → (batch, output_shape)
        out = self.nonlinear_TDoA(combined)

        # Safe squeeze: only removes the last dim if it's size 1, never touches batch dim
        if out.shape[-1] == 1:
            out = out.squeeze(-1)

        if return_unnormalized:
            out = (out * self.label_std) + self.label_mean

        return out

class RNO_four_gentle_non_linear_merge_norm_cnn(nn.Module):
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
                 label_mean: ndarray | None | torch.Tensor = None,
                 label_std: ndarray | None | torch.Tensor = None,
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
        self.register_buffer('label_mean', torch.as_tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std', torch.as_tensor(label_std, dtype=torch.float32))
        
        # --- STATION BLOCK (Feature Extraction) ---
        self.station_block = nn.Sequential(
            # 1. Normalize raw voltages per-channel before any convolution
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True),

            # 2. Layer 1: Detect pulses at FULL time resolution (1024 bins = 8192 ns)
            # Wide time kernel to capture waveform shape before any downsampling
            nn.Conv3d(in_channels=input_shape, out_channels=hidden_units,
                    kernel_size=(3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(hidden_units, momentum=0.01, affine=True),
            nn.LeakyReLU(negative_slope=leak_factor),
            # Pool 4x in time ONLY — 1024 → 256 bins (32 ns resolution preserved)
            # Physically justified: max meaningful TDoA ~50 bins, 4x pool keeps ~12 bins of precision
            # Reduces activation size ~4x: 65GB → ~16GB
            # Station and antenna dimensions untouched — spatial locality preserved
            nn.MaxPool3d(kernel_size=(1,4,1)),

            # 3. Layer 2: Refine features at 256 time bins — sufficient TDoA resolution
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units,
                    kernel_size=(3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(hidden_units, momentum=0.01, affine=True),
            nn.LeakyReLU(negative_slope=leak_factor),

            # 4. Layer 3: Deep abstract features — still at 256 time bins
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units,
                    kernel_size=(3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(hidden_units, momentum=0.01, affine=True),
            nn.LeakyReLU(negative_slope=leak_factor),
        )

        # --- NORMALIZATION BLOCK ---
        # Compresses hidden_units channels down to 1 for the TDoA block.
        # Two-step compression instead of one aggressive step — gentler gradient flow.
        # 64 → 16 → 1 rather than 64 → 1 in a single 1x1 conv.
        self.normalization_block = nn.Sequential(
            # Step 1: compress to hidden_units//4 channels (e.g. 64 → 16)
            nn.Conv3d(in_channels=hidden_units, out_channels=hidden_units//4, kernel_size=1),
            nn.LeakyReLU(negative_slope=leak_factor),
            # Step 2: compress to 1 channel (e.g. 16 → 1)
            nn.Conv3d(in_channels=hidden_units//4, out_channels=1, kernel_size=1),
            nn.LeakyReLU(negative_slope=leak_factor),
            # Final spatial projection to fixed size for the linear TDoA block
            nn.AdaptiveMaxPool3d((24, temporal_res, station_num))
        )

        # --- LINEAR TDOA BLOCK ---
        # Input: 1 channel × 24 antennas × temporal_res bins × station_num stations
        self.flatten_size = 1 * 24 * temporal_res * station_num

        self.nonlinear_TDoA = nn.Sequential(
            nn.Flatten(),

            # Layer 1: Cross-station feature compression
            nn.Linear(in_features=self.flatten_size, out_features=256),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Layer 2: Geometric reasoning — learns TDoA triangulation relationships
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(p=dropout_rate),

            # Output: normalized (x, y, z) vertex coordinates
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
                x = (x * self.label_std) + self.label_mean #type: ignore
                
            return x


class RNO_four_branch_cnn(nn.Module):
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
                 label_mean: ndarray | None | torch.Tensor = None,
                 label_std: ndarray | None | torch.Tensor = None,
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
        self.register_buffer('label_mean', torch.as_tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std', torch.as_tensor(label_std, dtype=torch.float32))
        
        self.station_time_cnn_block = nn.Sequential(auto_name(
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True),

            # Large kernel, full resolution, padding to capture edges!
            nn.Conv3d(input_shape, hidden_units, kernel_size=(1,7,1), padding=(0,3,0), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            # Strided downsampling with growing channels
            nn.Conv3d(hidden_units, hidden_units, kernel_size=(1,5,1), stride=(1,2,1), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            nn.Conv3d(hidden_units, hidden_units, kernel_size=(1,3,1), stride=(1,2,1), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.LeakyReLU(negative_slope=leak_factor),

            nn.Conv3d(hidden_units, hidden_units, kernel_size=(1,3,1), stride=(1,2,1), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            nn.Conv3d(hidden_units, hidden_units, kernel_size=(1,3,1), stride=(1,2,1), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            nn.Dropout(dropout_rate),
        ))

        self.station_channel_cnn_block = nn.Sequential(auto_name(
            # Span antennas — learns local beamforming patterns
            nn.Conv3d(hidden_units, hidden_units, kernel_size=(5,5,1), padding=(2,0,0), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            # Span more antennas + compress time further
            nn.Conv3d(hidden_units, hidden_units, kernel_size=(3,5,1), stride=(2,4,1), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            # Span more antennas + compress time further
            nn.Conv3d(hidden_units, hidden_units, kernel_size=(3,3,1), stride=(2,2,1), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            # Compress channels aggressively now that spatial is small
            nn.Conv3d(hidden_units, hidden_units//2, kernel_size=(1,1,1), bias=False),
            nn.BatchNorm3d(hidden_units//2, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            nn.Conv3d(hidden_units//2, hidden_units//4, kernel_size=(1,1,1), bias=False),
            nn.BatchNorm3d(hidden_units//4, momentum=0.01),
            nn.LeakyReLU(negative_slope=leak_factor),

            nn.Dropout(dropout_rate),
            nn.Flatten(),
        ))

        with torch.no_grad(): # no_grad saves memory during this check
                    dummy_input = torch.zeros(1, input_shape, 24, 1024, 4)
                    dummy_out = self.station_time_cnn_block(dummy_input)
                    dummy_out = self.station_channel_cnn_block(dummy_out)
                    flattened_size = dummy_out.shape[1]

        # LazyLinear handles the flatten size automatically
        self.station_ToA_block = nn.Sequential(auto_name(
            nn.Linear(flattened_size,128),   
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=leak_factor),
            nn.Linear(32, output_shape),
        ))

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
            x = self.station_time_cnn_block(x)
            x = self.station_channel_cnn_block(x)
            x = self.station_ToA_block(x)
            # Ensure the output is shape (Batch, 3)
            x = torch.squeeze(x)
            
            # If we are in final evaluation mode, convert the network's 
            # [-1, 1] normalized output back into real-world physical units.
            if return_unnormalized:
                # Broadcasts the math perfectly across the batch dimension
                x = (x * self.label_std) + self.label_mean #type: ignore
                
            return x

class ResBlock3d(nn.Module):
    """
    3D ResNet block with identity skip connection.
    Input and output channels must be equal (no projection needed).
    BN → Conv → BN → ReLU → Conv → BN → add residual → ReLU
    """
    def __init__(self, channels: int, kernel_size: tuple, padding: tuple,
                 stride: tuple = (1,1,1)):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm3d(channels, momentum=0.01)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                               padding=padding, bias=False)
        self.bn2   = nn.BatchNorm3d(channels, momentum=0.01)
        self.act   = nn.ReLU()

        # If stride reduces spatial dims, project the skip connection to match
        self.downsample = None
        if any(s != 1 for s in stride):
            self.downsample = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels, momentum=0.01)
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)

class RNO_four_branch_resnet(nn.Module):
    """
    ResNet variant of RNO_four_branch_cnn.

    Same two-stage design — temporal extraction then cross-antenna fusion —
    but each conv layer is replaced by a ResNet block with an identity skip
    connection. This gives gradients a direct backward path through every
    block, preventing the vanishing gradient problem that killed deeper
    conv layers in earlier architectures.

    Stage 1 — Temporal ResNet (station_time_resnet_block):
        ResNet blocks with kernel (1, T, 1) — time axis only.
        Strided blocks downsample time progressively 1024 → ~62 bins.

    Stage 2 — Antenna fusion ResNet (station_channel_resnet_block):
        ResNet blocks with kernel (3, T, 1) — spans 3 antennas.
        Learns cross-antenna TDoA correlations with gradient highways.

    Stage 3 — TDoA regression head (station_ToA_block):
        Small linear head on the flattened, compressed feature maps.

    Input:  (batch, 1, 24, 1024, 4)
    Output: (batch, 3) — normalized (x, y, z)
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 label_mean: ndarray | None | torch.Tensor = None,
                 label_std:  ndarray | None | torch.Tensor = None,
                 station_num: int = 4,
                 leak_factor: float = 0.1,
                 dropout_rate: float = 0.1,
                 temporal_res: int = 128):

        super().__init__()

        if label_mean is None: label_mean = torch.zeros(output_shape)
        if label_std  is None: label_std  = torch.ones(output_shape)
        self.register_buffer('label_mean', torch.as_tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std',  torch.as_tensor(label_std,  dtype=torch.float32))

        lf = leak_factor

        # ----------------------------------------------------------------
        # STAGE 1: Temporal feature extraction
        # First conv lifts input_shape → hidden_units (not a ResBlock since
        # channels change). All subsequent blocks are identity ResBlocks.
        # ----------------------------------------------------------------
        self.station_time_resnet_block = nn.Sequential(
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True),

            # Channel lifting conv — not a ResBlock (in ≠ out channels)
            nn.Conv3d(input_shape, hidden_units, kernel_size=(1,7,1),
                      padding=(0,3,0), bias=False),
            nn.BatchNorm3d(hidden_units, momentum=0.01),
            nn.ReLU(),

            ResBlock3d(hidden_units, kernel_size=(1,5,1), padding=(0,2,0),
                       stride=(1,4,1)),   # 256→128

            ResBlock3d(hidden_units, kernel_size=(1,5,1), padding=(0,2,0),
                       stride=(1,4,1)),   # 128→64

            nn.Dropout(dropout_rate),
        )

        # ----------------------------------------------------------------
        # STAGE 2: Cross-antenna fusion
        # kernel=(3,T,1) spans 3 antennas. padding=(1,X,0) preserves
        # antenna dimension so edge antennas are weighted equally.
        # Strided blocks compress antennas and time together.
        # ----------------------------------------------------------------
        self.station_channel_resnet_block = nn.Sequential(

            # Non-strided block — antenna mixing at full resolution
            ResBlock3d(hidden_units, kernel_size=(3,5,1), padding=(1,2,0)),

            # Strided blocks — compress antennas and time simultaneously
            ResBlock3d(hidden_units, kernel_size=(3,3,1), padding=(1,1,0),
                       stride=(2,4,1)),   # 24→12, 64→32
            
            ResBlock3d(hidden_units, kernel_size=(3,3,1), padding=(1,1,0),
                                stride=(2,2,1)),   # 12→6, 16→8

            # Channel compression via 1×1 convs — no skip needed, small
            nn.Conv3d(hidden_units, hidden_units//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_units//2, momentum=0.01),
            nn.ReLU(),

            nn.Conv3d(hidden_units//2, hidden_units//4, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_units//4, momentum=0.01),
            nn.ReLU(),

            nn.Dropout(dropout_rate),
            nn.Flatten(),
        )

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, 24, 1024, 4)
            dummy = self.station_time_resnet_block(dummy)
            dummy = self.station_channel_resnet_block(dummy)
            flattened_size = dummy.shape[1]

        print(f"Flattened size: {flattened_size}")

        # ----------------------------------------------------------------
        # STAGE 3: Regression head
        # ----------------------------------------------------------------
        self.station_ToA_block = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape),
        )

    def forward(self, x: torch.Tensor, return_unnormalized: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input voltages, shape (batch, 1, 24, 1024, 4).
            return_unnormalized: If True, converts output to physical
                coordinates (meters). Use at inference only.
        """
        x = self.station_time_resnet_block(x)
        x = self.station_channel_resnet_block(x)
        x = self.station_ToA_block(x)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        if return_unnormalized:
            x = (x * self.label_std) + self.label_mean  # type: ignore

        return x