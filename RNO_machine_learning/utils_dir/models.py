import torch
from torch import nn
from numpy import ndarray

class RNO_four_just_linear_merge(nn.Module):
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

        # flatten_size
        self.flatten_size = 1 * 24 * 1024 * 4

        # Linear Layer
        self.linear = nn.Sequential(
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True), 
            nn.Flatten(),
            nn.Linear(self.flatten_size,output_shape)
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
            x = self.linear(x)
            
            # Ensure the output is shape (Batch, 3)
            x = torch.squeeze(x)
            
            # If we are in final evaluation mode, convert the network's 
            # [-1, 1] normalized output back into real-world physical units.
            if return_unnormalized:
                # Broadcasts the math perfectly across the batch dimension
                x = (x * self.label_std) + self.label_mean
                
            return x

class RNO_four_just_non_linear_merge(nn.Module):
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

        # --- LINEAR TDOA BLOCK ---
        # Channels (hidden) * Depth (24) * Time (temporal_res) * Stations (station_num)
        self.flatten_size = 1 * 24 * 1024 * station_num
        
        self.nonlinear_TDoA = nn.Sequential(
            nn.BatchNorm3d(input_shape, momentum=0.01, affine=True), 
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
            x = self.nonlinear_TDoA(x)
            
            # Ensure the output is shape (Batch, 3)
            x = torch.squeeze(x)
            
            # If we are in final evaluation mode, convert the network's 
            # [-1, 1] normalized output back into real-world physical units.
            if return_unnormalized:
                # Broadcasts the math perfectly across the batch dimension
                x = (x * self.label_std) + self.label_mean
                
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