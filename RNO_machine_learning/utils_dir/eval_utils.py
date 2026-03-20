"""
eval_utils.py
-------------
Evaluation utilities for RNO-G vertex reconstruction models.

The ModelEvaluator class handles the full evaluation pipeline:
loading a checkpoint, running inference on a validation dataset,
computing physical metrics, and saving results for comparison.

The from_checkpoint classmethod is the preferred entry point for
new checkpoints — it reads all model and dataset information directly
from the checkpoint file, eliminating manual specification and human error.

Author: Santiago Sued
"""

from torch.utils.data import DataLoader
from .dataset import ShardStreamIterableDataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import warnings
import torch
import json
import glob


class ModelEvaluator:
    """
    Evaluates a trained RNO-G vertex reconstruction model and stores the results.
    Automatically runs the evaluation pipeline upon instantiation.

    Preferred usage for new checkpoints (saved with model_config):
        evaluator = ModelEvaluator.from_checkpoint('/path/to/checkpoints')

    Legacy usage for old checkpoints (no model_config):
        evaluator = ModelEvaluator(model=model, model_checkpoint=ckpt_dir, ...)
    """

    def __init__(self, model: torch.nn.Module, model_checkpoint: str | Path,
                 eval_dset: ShardStreamIterableDataset, batch_size: int = 256,
                 spherical: bool = True, load_npz: str | Path | None = None,
                 device=None):
        # Store configuration
        self.model            = model
        self.model_checkpoint = Path(model_checkpoint)
        self.eval_dset        = eval_dset
        self.batch_size       = batch_size
        self.spherical        = spherical
        self.device           = device

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

        # Device Setup
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Executing evaluation on: {self.device}")

        print(f"Loading evaluation dataset from: {self.eval_dset}")
        use_pin_memory = torch.cuda.is_available()
        eval_dloader = DataLoader(
            self.eval_dset,
            batch_size=None,   # dataset yields pre-assembled batches
            shuffle=False,
            pin_memory=use_pin_memory,
            num_workers=4
        )
        print(f"Dataset loaded. Total batches to evaluate: {len(eval_dloader)}")

        # Checkpoint Discovery
        if self.model_checkpoint.is_dir():
            print(f"Checkpoint path is a directory: {self.model_checkpoint}")
            print("Searching for the minimum test loss checkpoint ('*min_e*')...")
            try:
                self.model_checkpoint = next(self.model_checkpoint.glob('*min_e*'))
            except StopIteration:
                raise FileNotFoundError(
                    f"CRITICAL: Could not find any file containing '*min_e*' in {self.model_checkpoint}"
                )

        # Checkpoint Loading
        print(f"Loading weights from: {self.model_checkpoint.name}...")
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(self._uncompile_keys(state_dict))
        self.model.to(self.device)
        print(f"Checkpoint successfully loaded into model on {self.device}!")

        mean = self.model.label_mean.detach().cpu().numpy()  # type: ignore
        std  = self.model.label_std.detach().cpu().numpy()   # type: ignore
        print('Trained unnormalized statistics (Loaded from Buffer):')
        print(f"  X = {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"  Y = {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"  Z = {mean[2]:.4f} ± {std[2]:.4f}")

        if self.true_arr is None:
            true_list = []
            reco_list = []
            print("Starting forward passes...")
            self.model.eval()

            with torch.inference_mode():
                for (image, true) in tqdm(eval_dloader, desc="Evaluating", unit='batch'):
                    image = image.to(self.device)
                    reco  = self.model(image, return_unnormalized=True)
                    true_list.append(true.cpu().numpy())
                    reco_list.append(reco.cpu().numpy())

            print("Inference complete. Formatting output arrays...")
            true_arr_cat = np.concatenate(true_list, axis=0)
            reco_arr_cat = np.concatenate(reco_list, axis=0)

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

    # ================================================================
    # METRICS
    # ================================================================

    def show_rel_R(self, hist: bool = False):
        """
        Calculates, prints, and returns the relative error of the reconstructed radius.
        Formula: (R_reco - R_true) / R_true

        Args:
            hist (bool): If True, plots a histogram of the relative error.

        Returns:
            np.ndarray: Per-event relative radius errors.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: Evaluation has not been run yet.")
            return None

        if not self.spherical:
            print("Converting internal arrays to Spherical coordinates to extract R...")
            self.true_arr = np.array([self.cart_2_spher(*row) for row in self.true_arr])
            self.reco_arr = np.array([self.cart_2_spher(*row) for row in self.reco_arr])
            self.spherical = True

        true_R = self.true_arr[:, 0]
        reco_R = self.reco_arr[:, 0]

        valid_mask = true_R != 0
        rel_R = np.zeros_like(true_R)
        rel_R[valid_mask] = (reco_R[valid_mask] - true_R[valid_mask]) / true_R[valid_mask]

        mean_rel   = float(np.mean(rel_R))
        median_rel = float(np.median(rel_R))
        std_rel    = float(np.std(rel_R))
        p16        = float(np.percentile(rel_R, 16))
        p84        = float(np.percentile(rel_R, 84))
        sigma_68   = (p84 - p16) / 2.0

        print("--- Relative Radius (R) Error Statistics ---")
        print(f"Mean Error:   {mean_rel:+.4f} ({mean_rel:+.2%})")
        print(f"Median Error: {median_rel:+.4f} ({median_rel:+.2%})")
        print(f"Spread (Std): {std_rel:.4f} ({std_rel:.2%})")
        print("-" * 44)
        print(f"Sigma 68%:    {sigma_68:.4f} ({sigma_68:.2%})")
        print(f"68% Interval: [{p16:+.4f}, {p84:+.4f}]")

        if hist:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 6))
            plt.hist(rel_R, bins=50, color='royalblue', alpha=0.7, edgecolor='black')
            plt.axvspan(p16, p84, color='mediumseagreen', alpha=0.25,
                        label=rf'68% Interval: [{p16:+.2f}, {p84:+.2f}] ($\sigma_{{68}}$ = {sigma_68:.3f})')
            plt.axvline(p16, color='seagreen', linestyle='dotted', linewidth=2)
            plt.axvline(p84, color='seagreen', linestyle='dotted', linewidth=2)
            plt.axvline(mean_rel, color='red', linestyle='dashed', linewidth=2,
                        label=f'Mean: {mean_rel:+.3f} ({mean_rel:+.2%})')
            plt.axvline(median_rel, color='darkorange', linestyle='dashed', linewidth=2,
                        label=f'Median: {median_rel:+.3f} ({median_rel:+.2%})')
            plt.title('Relative Error of Reconstructed Radius (R)')
            plt.xlabel('Relative Error: (R_reco - R_true) / R_true')
            plt.ylabel('Number of Events')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        return rel_R
    
    def show_euclidean_distance(self, hist: bool = False):
        """
        Calculates, prints, and returns the 3D Euclidean distance error in meters.
        Formula: sqrt((x_reco - x_true)^2 + (y_reco - y_true)^2 + (z_reco - z_true)^2)

        Distance error is always positive, so resolution is reported as the 68%
        upper containment (p84) rather than a symmetric interval — p84 means
        "68% of events are reconstructed within this many meters."

        Args:
            hist (bool): If True, plots a histogram of the distance errors.

        Returns:
            np.ndarray: Per-event 3D distance errors in meters.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: Evaluation has not been run yet.")
            return None

        # Convert to Cartesian if needed — distance is Cartesian by definition
        if self.spherical:
            true_cart = np.array([self.spher_2_cart(*row) for row in self.true_arr])
            reco_cart = np.array([self.spher_2_cart(*row) for row in self.reco_arr])
        else:
            true_cart = self.true_arr
            reco_cart = self.reco_arr

        dist = np.sqrt(((reco_cart - true_cart)**2).sum(axis=1))

        mean_d   = float(np.mean(dist))
        median_d = float(np.median(dist))
        std_d    = float(np.std(dist))
        p68      = float(np.percentile(dist, 68))   # 68% upper containment

        # Mode computed via histogram — np has no direct mode for continuous data
        counts, bin_edges = np.histogram(dist, bins=100)
        mode_d = float((bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1]) / 2)

        print("--- 3D Euclidean Distance Error (meters) ---")
        print(f"Mode:              {mode_d:.1f} m  (most common error)")
        print(f"Median:            {median_d:.1f} m")
        print(f"Mean:              {mean_d:.1f} m")
        print(f"Std:               {std_d:.1f} m")
        print("-" * 44)
        # p68 = 68% upper containment: 68% of events reconstructed within this distance
        print(f"68% containment:   {p68:.1f} m  (68% of events within this distance)")

        if hist:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 6))
            plt.hist(dist, bins=50, color='tomato', alpha=0.7, edgecolor='black')
            plt.axvline(mode_d,   color='purple',    linestyle='dashed', linewidth=2,
                        label=f'Mode:   {mode_d:.0f} m')
            plt.axvline(median_d, color='darkorange', linestyle='dashed', linewidth=2,
                        label=f'Median: {median_d:.0f} m')
            plt.axvline(mean_d,   color='red',        linestyle='dashed', linewidth=2,
                        label=f'Mean:   {mean_d:.0f} m')
            plt.axvline(p68,      color='seagreen',   linestyle='dotted', linewidth=2,
                        label=f'68% containment: {p68:.0f} m')
            plt.axvspan(0, p68, color='mediumseagreen', alpha=0.15)
            plt.title('3D Euclidean Distance Error')
            plt.xlabel('Distance Error (m)')
            plt.ylabel('Number of Events')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        return dist

    def save_to_log(self, experiment_name: str | None = None,
                    log_path: str | Path = 'eval_log.json'):
        """
        Appends this model's evaluation metrics to a shared JSON log file.
        If the file doesn't exist, creates it. If the experiment already has
        an entry, it is overwritten — re-evaluating a model updates its record
        rather than duplicating it. Other models' entries are never touched.

        Args:
            experiment_name: Key for this entry. If None, automatically derived
                            from the checkpoint's parent directory name — which
                            is the WandB run ID when using the standard directory
                            structure. Override with a custom name if needed.
            log_path:        Path to the shared JSON log file.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: No results to save. Run evaluation first.")
            return

        # Derive experiment name from checkpoint path if not provided
        # Structure: .../experiments/<run_id>/checkpoints/checkpoint_min_eX.pth
        # Parent of checkpoint file = checkpoints dir
        # Parent of that = run_id dir
        if experiment_name is None:
            experiment_name = self.model_checkpoint.parent.parent.name
            print(f"No experiment name provided — using: {experiment_name}")

        log_path = Path(log_path)

        # --- Relative radius metrics ---
        if self.spherical:
            true_R = self.true_arr[:, 0]
            reco_R = self.reco_arr[:, 0]
            valid  = true_R != 0
            rel_R  = np.zeros_like(true_R)
            rel_R[valid] = (reco_R[valid] - true_R[valid]) / true_R[valid]
            p16_R, p84_R  = np.percentile(rel_R, [16, 84])
            rel_R_median  = np.median(rel_R)
        else:
            p16_R = p84_R = float('nan')
            rel_R_median  = float('nan')

        # --- Distance metrics ---
        true_cart = np.array([self.spher_2_cart(*r) for r in self.true_arr]) if self.spherical else self.true_arr
        reco_cart = np.array([self.spher_2_cart(*r) for r in self.reco_arr]) if self.spherical else self.reco_arr
        dist      = np.sqrt(((reco_cart - true_cart)**2).sum(axis=1))
        p68_d     = float(np.percentile(dist, 68))

        counts, bin_edges = np.histogram(dist, bins=100)
        mode_d = float((bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1]) / 2)

        # --- Epoch and notes from checkpoint ---
        ckpt             = torch.load(self.model_checkpoint, map_location='cpu')
        checkpoint_epoch = int(ckpt.get('epoch', -1))
        notes            = ckpt.get('model_config', {}).get('notes', None)

        entry = {
            # Relative radius
            'rel_R_median'       : round(float(rel_R_median), 4),
            'rel_R_p16'          : round(float(p16_R), 4),
            'rel_R_p84'          : round(float(p84_R), 4),
            'rel_R_sigma68'      : round(float((p84_R - p16_R) / 2), 4),
            # Distance
            'dist_mode_m'        : round(mode_d, 2),
            'dist_median_m'      : round(float(np.median(dist)), 2),
            'dist_68pct_contain' : round(p68_d, 2),
            # Metadata
            'n_events'           : int(len(dist)),
            'checkpoint_file'    : str(self.model_checkpoint.name),
            'checkpoint_epoch'   : checkpoint_epoch,
            'notes'              : notes,
            'evaluated_at'       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Load existing log or start fresh — never overwrites other experiments
        log = {}
        if log_path.exists():
            with open(log_path, 'r') as f:
                log = json.load(f)

        log[experiment_name] = entry

        with open(log_path, 'w') as f:
            json.dump(log, f, indent=4)

        print(f"\n{'='*60}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Notes      : {notes or 'None'}")
        print(f"  Epoch      : {checkpoint_epoch}")
        print(f"{'='*60}")
        print(f"  rel_R    median={entry['rel_R_median']:+.4f}  68%: [{entry['rel_R_p16']:+.4f}, {entry['rel_R_p84']:+.4f}]  σ68={entry['rel_R_sigma68']:.4f}")
        print(f"  dist     mode={entry['dist_mode_m']:.1f}m  median={entry['dist_median_m']:.1f}m  68% containment={entry['dist_68pct_contain']:.1f}m")
        print(f"  Saved to : {log_path}")
        print(f"{'='*60}\n")
    
    def plot_3d_displacement(self, n_points: int = 50, seed: int = 42):
        """
        Plots a 3D scatter of a subset of events, drawing a line between
        the True vertex and the Reconstructed vertex to visualize error vectors.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: Evaluation has not been run yet.")
            return

        import matplotlib.pyplot as plt

        if self.spherical:
            print("Temporarily converting a subset back to Cartesian for 3D plotting...")
            true_data = np.array([self.spher_2_cart(*row) for row in self.true_arr])
            reco_data = np.array([self.spher_2_cart(*row) for row in self.reco_arr])
        else:
            true_data = self.true_arr
            reco_data = self.reco_arr

        np.random.seed(seed)
        n_points = min(n_points, len(true_data))
        indices  = np.random.choice(len(true_data), n_points, replace=False)
        t_sample = true_data[indices]
        r_sample = reco_data[indices]

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        t_x, t_y, t_z = t_sample[:, 0], t_sample[:, 1], t_sample[:, 2]
        r_x, r_y, r_z = r_sample[:, 0], r_sample[:, 1], r_sample[:, 2]

        ax.scatter(t_x, t_y, t_z, c='mediumseagreen', marker='o', s=40, label='True Vertex', alpha=0.8)   # type: ignore
        ax.scatter(r_x, r_y, r_z, c='tomato',         marker='X', s=40, label='Reco Vertex', alpha=0.8)   # type: ignore

        for i in range(n_points):
            ax.plot([t_x[i], r_x[i]], [t_y[i], r_y[i]], [t_z[i], r_z[i]],
                    color='gray', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_title(f'3D Vertex Displacement (Subset of {n_points} events)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str | Path = "eval_results.npz"):
        """Saves the evaluation arrays to a compressed NumPy file."""
        if self.true_arr is None or self.reco_arr is None:
            print("Error: No data to save.")
            return
        filename = Path(filename)
        np.savez_compressed(filename, true_arr=self.true_arr,
                            reco_arr=self.reco_arr, spherical=self.spherical)
        print(f"Saved to {filename}")

    def load_results(self, filename: str | Path = "eval_results.npz"):
        """Loads previously saved evaluation arrays."""
        filename = Path(filename)
        if not filename.exists():
            print(f"Error: Could not find {filename}")
            return
        with np.load(filename) as data:
            self.true_arr  = data['true_arr']
            self.reco_arr  = data['reco_arr']
            self.spherical = data['spherical'].item()
        print(f"Loaded from {filename}")
        print(f"  True array shape: {self.true_arr.shape}")
        print(f"  Spherical coords: {self.spherical}")

    # ================================================================
    # STATIC / CLASS METHODS
    # ================================================================

    @staticmethod
    def cart_2_spher(x, y, z):
        """Converts Cartesian (x, y, z) to Spherical (r, theta, phi)."""
        r = np.sqrt(x**2 + y**2 + z**2)
        θ = np.arccos(z / r) if np.any(r != 0) else 0
        φ = np.arctan2(y, x)
        return r, θ, φ

    @staticmethod
    def spher_2_cart(r, theta, phi):
        """Converts Spherical (r, theta, phi) back to Cartesian (x, y, z)."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def _uncompile_keys(state_dict: dict) -> dict:
        """Removes torch.compile prefix '_orig_mod.' from state dict keys."""
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, batch_size: int = 512,
                        spherical: bool = True, device: str | None = None):
        """
        Alternative constructor that reads model architecture and dataset
        information directly from the checkpoint's saved model_config.

        Only works for checkpoints saved after model_config was added to
        save_checkpoint. Old checkpoints raise a ValueError with a clear
        message pointing to the standard constructor as the fallback.

        Args:
            checkpoint_path: Path to a checkpoint file or checkpoints directory.
                             If a directory, automatically finds the best checkpoint.
            batch_size:      Inference batch size.
            spherical:       If True, converts output coordinates to spherical.
            device:          Device to run inference on (e.g. 'cuda:1').
                             Defaults to cuda if available, otherwise cpu.

        Returns:
            ModelEvaluator: Fully initialized and evaluated instance.
        """
        import importlib

        checkpoint_path = Path(checkpoint_path)

        # Resolve directory to best checkpoint file
        if checkpoint_path.is_dir():
            matches = list(checkpoint_path.glob('*min_e*'))
            if matches:
                checkpoint_path = matches[0]
            else:
                warnings.warn(f"No 'min_e' checkpoint found in {checkpoint_path}. Falling back to the latest checkpoint.")
                all_checkpoints = [p for p in checkpoint_path.iterdir() if p.is_file()]

                if not all_checkpoints:
                    raise FileNotFoundError(f"No checkpoints found at all in {checkpoint_path}")
                
                checkpoint_path = max(all_checkpoints, key=lambda p: p.stat().st_mtime)

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg  = ckpt.get('model_config')

        if cfg is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path.name} has no model_config — "
                f"it was saved before this feature was added. "
                f"Use the standard ModelEvaluator constructor instead."
            )

        # Dynamically resolve model class by name — no hardcoded registry needed.
        # As long as the class is importable from utils_dir.models, it just works.
        models_module = importlib.import_module('utils_dir.models')
        model_class   = getattr(models_module, cfg['model_class'])

        model = model_class(
            input_shape  = cfg['input_shape'],
            output_shape = cfg['output_shape'],
            hidden_units = cfg['hidden_units'],
            leak_factor  = cfg.get('leak_factor',  0.1),
            dropout_rate = cfg.get('dropout_rate', 0.0),
            temporal_res = cfg.get('temporal_res', 256),
        )

        # Build val dataset from the saved manifest path.
        # No label stats are passed — the dataset yields unnormalized labels
        # when stats are None, which is correct since return_unnormalized=True
        # is used during inference.
        with open(cfg['manifest_path']) as f:
            manifest = json.load(f)
        val_paths = manifest['splits']['val']['files']

        val_dset = ShardStreamIterableDataset(
            shard_file_list  = val_paths,
            manifest_path    = cfg['manifest_path'],
            batch_size       = batch_size,
            is_train         = False,
            min_station_hits = cfg.get('min_station_hits', 1),
        )

        # Load notes and experiment name from checkpoint
        notes           = cfg.get('notes', None)
        experiment_name = checkpoint_path.parent.parent.name  # <run_id>

        print(f"\n{'='*60}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Notes      : {notes or 'None'}")
        print(f"{'='*60}")
        print(f"  Model    : {cfg['model_class']} | hidden_units={cfg['hidden_units']} | temporal_res={cfg.get('temporal_res', 256)}")
        print(f"  Manifest : {cfg['manifest_path']}")
        print(f"  Training : lr={cfg.get('learning_rate')} | wd={cfg.get('weight_decay')} | min_hits={cfg.get('min_station_hits', 1)}")
        print(f"{'='*60}\n")

        return cls(
            model            = model,
            model_checkpoint = checkpoint_path,
            eval_dset        = val_dset,
            batch_size       = batch_size,
            spherical        = spherical,
            device           = device,
        )
    
    def show_error_vs_radius(self, n_bins: int = 10):
        """
        Plots mean absolute relative radius error as a function of true radius.
        Reveals whether the model performs differently at different distances —
        the shrinkage bias often worsens for large-radius (distant) events.

        Args:
            n_bins: Number of radius bins to divide events into.
        """
        if self.true_arr is None or self.reco_arr is None:
            print("Error: Evaluation has not been run yet.")
            return

        import matplotlib.pyplot as plt

        # Work in spherical — R is column 0
        true_R = self.true_arr[:, 0]
        reco_R = self.reco_arr[:, 0]

        valid  = true_R != 0
        rel_R  = np.zeros_like(true_R)
        rel_R[valid] = (reco_R[valid] - true_R[valid]) / true_R[valid]

        # Also compute 3D distance error in Cartesian
        true_cart = np.array([self.spher_2_cart(*r) for r in self.true_arr])
        reco_cart = np.array([self.spher_2_cart(*r) for r in self.reco_arr])
        dist_err  = np.sqrt(((reco_cart - true_cart)**2).sum(axis=1))

        # Bin events by true radius
        bin_edges  = np.percentile(true_R, np.linspace(0, 100, n_bins + 1))
        bin_centers, mean_rel_R, mean_dist, median_dist = [], [], [], []

        for i in range(n_bins):
            mask = (true_R >= bin_edges[i]) & (true_R < bin_edges[i+1])
            if mask.sum() == 0:
                continue
            bin_centers.append(float(np.median(true_R[mask])))
            mean_rel_R.append(float(np.mean(rel_R[mask])))
            mean_dist.append(float(np.mean(dist_err[mask])))
            median_dist.append(float(np.median(dist_err[mask])))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: mean relative radius error vs true radius
        ax1.plot(bin_centers, mean_rel_R, 'o-', color='royalblue', linewidth=2)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.fill_between(bin_centers, mean_rel_R, 0,
                        alpha=0.2, color='royalblue')
        ax1.set_xlabel('True Radius (m)')
        ax1.set_ylabel('Mean Relative Radius Error')
        ax1.set_title('Relative Radius Error vs True Radius\n(negative = undershooting)')
        ax1.grid(alpha=0.3)

        # Plot 2: mean/median euclidean distance error vs true radius
        ax2.plot(bin_centers, mean_dist,   'o-', color='tomato',     linewidth=2, label='Mean')
        ax2.plot(bin_centers, median_dist, 's-', color='darkorange',  linewidth=2, label='Median')
        ax2.set_xlabel('True Radius (m)')
        ax2.set_ylabel('Distance Error (m)')
        ax2.set_title('3D Distance Error vs True Radius')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary table
        print(f"\n{'True R bin (m)':<20} {'Mean rel_R':>12} {'Mean dist (m)':>15} {'Median dist (m)':>16}")
        print('-' * 65)
        for i in range(len(bin_centers)):
            print(f"{bin_centers[i]:<20.0f} {mean_rel_R[i]:>+12.4f} {mean_dist[i]:>15.1f} {median_dist[i]:>16.1f}")

def get_experiment_checkpoints(EXPERIMENTS_DIR):
    """
    Locates checkpoint directories for all experiments within a root directory.
    
    This function scans subdirectories of the provided root path for a folder 
    named 'checkpoints' and maps the experiment's name to its full path.

    Args:
        EXPERIMENTS_DIR (str): The base directory where experiments are stored.

    Returns:
        dict: A dictionary where keys are experiment names (folder names) 
              and values are the full string paths to their 'checkpoints' folder.
    """
    # Find all directories matching the pattern: Root/Experiment_Name/checkpoints
    experiment_checkpoint_paths = [Path(path) for path in glob.glob(EXPERIMENTS_DIR + '/*/checkpoints')]
    
    # Extract the name of the experiment (the parent folder of the 'checkpoints' dir)
    experiment_names = [str(experiment_path.parent.name) for experiment_path in experiment_checkpoint_paths]
    
    # Convert Path objects back to strings for the final dictionary values
    experiment_checkpoint_paths = [str(experiment_checkpoint_path) for experiment_checkpoint_path in experiment_checkpoint_paths]
    
    # Pair names with their paths into a searchable dictionary
    return dict(zip(experiment_names, experiment_checkpoint_paths))