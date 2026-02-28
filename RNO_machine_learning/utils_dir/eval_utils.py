from torch.utils.data import DataLoader
from .dataset import ShardAlbumDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

class ModelEvaluator:
    """
    Evaluates a trained RNO-G vertex reconstruction model and stores the results.
    Automatically runs the evaluation pipeline upon instantiation.
    """
    
    def __init__(self, model: torch.nn.Module, model_checkpoint: str | Path, eval_dset: str | Path, batch_size: int = 256, spherical: bool = True, load_npz: str | Path | None = None):
        # Store configuration
        self.model = model
        self.model_checkpoint = Path(model_checkpoint)
        self.eval_dset = Path(eval_dset)
        self.batch_size = batch_size
        self.spherical = spherical
        
        # Initialize the data fields where the results will live
        self.true_arr = None
        self.reco_arr = None
        
        # Trigger the evaluation automatically unless load file is specified
        if load_npz:
            print("--- Skipping Evaluation ---")
            print(f" Loading from {load_npz}")
            self.load_results(load_npz)
            print("Data Loaded!")

        self._run_evaluation()

    def _run_evaluation(self):
        """Internal method that executes the evaluation pipeline."""
        print("--- Starting Evaluation Pipeline ---")
        
        # --- Device Setup ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Executing evaluation on: {device}")
        
        print(f"Loading evaluation dataset from: {self.eval_dset}")
        eval_album = ShardAlbumDataset(self.eval_dset)
         
        # --- Added pin_memory=True if using CUDA to dramatically speed up data transfers ---
        use_pin_memory = torch.cuda.is_available()
        eval_dloader = DataLoader(
            eval_album, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=use_pin_memory,
            num_workers=4 # Adjust to 1 or 2 if Jupyter notebook hangs
        )
        print(f"Dataset loaded. Total batches to evaluate: {len(eval_dloader)}")

        # Checkpoint Discovery
        if self.model_checkpoint.is_dir():
            print(f"Checkpoint path is a directory: {self.model_checkpoint}")
            print("Searching for the minimum test loss checkpoint ('*emin*')...")
            try:
                self.model_checkpoint = next(self.model_checkpoint.glob('*emin*'))
            except StopIteration:
                raise FileNotFoundError(f"CRITICAL: Could not find any file containing '*emin*' in {self.model_checkpoint}")

        # Checkpoint Loading
        print(f"Loading weights from: {self.model_checkpoint.name}...")
        
        # --- Load the checkpoint directly to the detected device ---
        checkpoint = torch.load(self.model_checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Load weights into model
        self.model.load_state_dict(self._uncompile_keys(state_dict))
        
        # --- Move the entire model to the GPU (or keep on CPU) ---
        self.model.to(device)
        print(f"Checkpoint successfully loaded into model on {device}!")

        # Extract stats (Your .cpu() calls here are perfect, they prevent crashes if model is on GPU)
        mean = self.model.label_mean.detach().cpu().numpy() #type: ignore
        std = self.model.label_std.detach().cpu().numpy() #type: ignore

        print('Trained unnormalized statistics (Loaded from Buffer):')
        print(f"  X = {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"  Y = {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"  Z = {mean[2]:.4f} ± {std[2]:.4f}")

        if self.true_arr is None:
            # Inference Loop
            true_list = []
            reco_list = []

            print("Starting forward passes...")
            self.model.eval() 

            with torch.inference_mode(): 
                for (image, true) in tqdm(eval_dloader, desc="Evaluating"):
                    # Data is moved to GPU here
                    image = image.to(device)
                    
                    reco = self.model(image, return_unnormalized=True)

                    # Your .cpu().numpy() calls here are already perfectly written for GPU!
                    true_list.append(true.cpu().numpy())
                    reco_list.append(reco.cpu().numpy())

            print("Inference complete. Formatting output arrays...")

            # Post-Processing
            true_arr_cat = np.concatenate(true_list, axis=0)
            reco_arr_cat = np.concatenate(reco_list, axis=0)

            # Coordinate Conversion
            if self.spherical:
                print("Converting Cartesian coordinates (X, Y, Z) to Spherical (R, Theta, Phi)...")
                self.true_arr = np.array([self.cart_2_spher(*row) for row in true_arr_cat])
                self.reco_arr = np.array([self.cart_2_spher(*row) for row in reco_arr_cat])
            else:
                print("Keeping Cartesian coordinates.")
                self.true_arr = true_arr_cat
                self.reco_arr = reco_arr_cat

            print("--- Evaluation Finished! ---")
            print(f"Shape of True array: {self.true_arr.shape}")
            print(f"Shape of Reco array: {self.reco_arr.shape}")
        else:
            print('--- Skipped Evaluation! Data already loaded! ---')

    def show_rel_R(self, hist: bool = False):
        """
        Calculates, prints, and returns the relative error of the reconstructed Radius (R).
        Formula: (R_reco - R_true) / R_true
        
        Args:
            hist (bool): If True, plots a histogram of the relative error.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: Evaluation has not been run yet. Data arrays are empty.")
            return None

        # Coordinate Check & Conversion
        if not self.spherical:
            print("Converting internal arrays to Spherical coordinates to extract R...")
            self.true_arr = np.array([self.cart_2_spher(*row) for row in self.true_arr])
            self.reco_arr = np.array([self.cart_2_spher(*row) for row in self.reco_arr])

            # Update the state flag so we don't accidentally convert twice
            self.spherical = True 

        # Extract Data (All rows, Column 0)
        true_R = self.true_arr[:, 0]
        reco_R = self.reco_arr[:, 0]

        # Calculate Relative Error safely (avoiding divide-by-zero if R=0)
        valid_mask = true_R != 0
        rel_R = np.zeros_like(true_R)
        rel_R[valid_mask] = (reco_R[valid_mask] - true_R[valid_mask]) / true_R[valid_mask]

        # Calculate Standard Summary Statistics
        mean_rel = float(np.mean(rel_R))
        median_rel = float(np.median(rel_R))
        std_rel = float(np.std(rel_R))

        # Calculate the 68% Containment Interval (Robust Sigma)
        p16 = float(np.percentile(rel_R, 16))
        p84 = float(np.percentile(rel_R, 84))
        sigma_68 = (p84 - p16) / 2.0

        # Print out the results beautifully formatted
        print("--- Relative Radius (R) Error Statistics ---")
        print(f"Mean Error:   {mean_rel:+.4f} ({mean_rel:+.2%}) (Average bias)")
        print(f"Median Error: {median_rel:+.4f} ({median_rel:+.2%}) (Typical bias)")
        print(f"Spread (Std): {std_rel:.4f} ({std_rel:.2%}) (Standard deviation, sensitive to outliers)")
        print("-" * 44)
        print(f"Sigma 68%:    {sigma_68:.4f} ({sigma_68:.2%}) (Robust resolution)")
        print(f"68% Interval: [{p16:+.4f}, {p84:+.4f}] (Contains central 68% of predictions)")
        
        # Plot histogram if requested
        if hist:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(9, 6))
            # Create a histogram with 50 bins
            plt.hist(rel_R, bins=50, color='royalblue', alpha=0.7, edgecolor='black')
            
            # Highlight the 68% containment region with a shaded span
            plt.axvspan(p16, p84, color='mediumseagreen', alpha=0.25, 
                        label=rf'68% Interval: [{p16:+.2f}, {p84:+.2f}] \n($\sigma_{{68}}$ = {sigma_68:.3f})')
            
            # Add dotted lines to clearly mark the 16th and 84th percentile bounds
            plt.axvline(p16, color='seagreen', linestyle='dotted', linewidth=2)
            plt.axvline(p84, color='seagreen', linestyle='dotted', linewidth=2)
            
            # Add vertical lines for the mean and median
            plt.axvline(mean_rel, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_rel:+.3f} ({mean_rel:+.2%})')
            plt.axvline(median_rel, color='darkorange', linestyle='dashed', linewidth=2, label=f'Median: {median_rel:+.3f} ({median_rel:+.2%})')
            
            # Format the plot
            plt.title('Relative Error of Reconstructed Radius (R)')
            plt.xlabel('Relative Error: (R_reco - R_true) / R_true')
            plt.ylabel('Number of Events')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout() # Ensures everything fits nicely
            plt.show()

        # Return the array just in case you want to do further analysis
        return rel_R

    def plot_3d_displacement(self, n_points: int = 50, seed: int = 42):
        """
        Plots a 3D scatter of a subset of events, drawing a line between 
        the True vertex and the Reconstructed vertex to visualize error vectors.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: Evaluation has not been run yet. Data arrays are empty.")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Coordinate Handling
        # We need Cartesian (X,Y,Z) to plot in 3D. 
        if self.spherical:
            print("Temporarily converting a subset back to Cartesian for 3D plotting...")
            true_data = np.array([self.spher_2_cart(*row) for row in self.true_arr])
            reco_data = np.array([self.spher_2_cart(*row) for row in self.reco_arr])
        else:
            true_data = self.true_arr
            reco_data = self.reco_arr

        # 2. Random Subsampling to avoid clutter
        np.random.seed(seed)
        total_events = len(true_data)
        # Ensure we don't try to sample more points than we have
        n_points = min(n_points, total_events) 
        
        indices = np.random.choice(total_events, n_points, replace=False)
        
        t_sample = true_data[indices]
        r_sample = reco_data[indices]

        # 3. Build the 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract axes for clean reading
        t_x, t_y, t_z = t_sample[:, 0], t_sample[:, 1], t_sample[:, 2]
        r_x, r_y, r_z = r_sample[:, 0], r_sample[:, 1], r_sample[:, 2]

        # Plot the scatter points
        ax.scatter(t_x, t_y, t_z, c='mediumseagreen', marker='o', s=40, label='True Vertex', alpha=0.8) #type: ignore
        ax.scatter(r_x, r_y, r_z, c='tomato', marker='X', s=40, label='Reco Vertex', alpha=0.8) #type: ignore

        # Plot the connecting displacement lines
        for i in range(n_points):
            ax.plot(
                [t_x[i], r_x[i]], 
                [t_y[i], r_y[i]], 
                [t_z[i], r_z[i]], 
                color='gray', linestyle='-', linewidth=1, alpha=0.5
            )

        # 4. Formatting
        ax.set_title(f'3D Vertex Displacement (Subset of {n_points} events)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str | Path = "eval_results.npz"):
        """
        Saves the evaluation arrays to a compressed NumPy file.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: No data to save. Run the evaluation first.")
            return
        
        filename = Path(filename)
        print(f"Saving results to {filename}...")
        
        # Pack the arrays and the state flag into a single compressed file
        np.savez_compressed(
            filename, 
            true_arr=self.true_arr, 
            reco_arr=self.reco_arr, 
            spherical=self.spherical
        )
        print("Save complete!")

    def load_results(self, filename: str | Path = "eval_results.npz"):
        """
        Loads previously saved evaluation arrays directly into the class instances.
        """
        filename = Path(filename)
        if not filename.exists():
            print(f"Error: Could not find {filename}")
            return
            
        print(f"Loading results from {filename}...")
        
        # Load the dictionary-like npz object
        with np.load(filename) as data:
            self.true_arr = data['true_arr']
            self.reco_arr = data['reco_arr']
            # .item() extracts the standard Python boolean from the 0-D numpy array
            self.spherical = data['spherical'].item() 
            
        print("Load complete!")
        print(f"Shape of loaded True array: {self.true_arr.shape}")
        print(f"Coordinates are currently Spherical: {self.spherical}")

    @staticmethod
    def cart_2_spher(x, y, z):
        """Helper static method to convert coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        θ  = np.arccos(z / r) if np.any(r != 0) else 0
        φ  = np.arctan2(y, x)
        return r, θ, φ

    @staticmethod
    def spher_2_cart(r, theta, phi):
        """Helper static method to convert Spherical back to Cartesian."""
        import numpy as np
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def _uncompile_keys(state_dict: dict) -> dict:
        """Helper static method to remove torch.compile prefix."""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "") 
            new_state_dict[new_key] = v
        return new_state_dict



    