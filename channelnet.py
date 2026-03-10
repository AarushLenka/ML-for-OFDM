import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# 1. Architecture Definition: The Dual-Stage ChannelNet Topology
# ==============================================================================
class OFDM_ChannelNet(nn.Module):
    def __init__(self):
        super(OFDM_ChannelNet, self).__init__()
        
        # Super-Resolution Block (SRCNN)
        self.sr_conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=9, padding=4)
        self.sr_prelu1 = nn.PReLU()
        self.sr_conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.sr_prelu2 = nn.PReLU()
        self.sr_conv3 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, padding=2)
        
        # Denoising Block (DnCNN)
        self.dn_conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.dn_prelu1 = nn.PReLU()
        self.dn_hidden = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.dn_conv_out = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        out_sr = self.sr_prelu1(self.sr_conv1(x))
        out_sr = self.sr_prelu2(self.sr_conv2(out_sr))
        out_sr = self.sr_conv3(out_sr)
        
        out_dn = self.dn_prelu1(self.dn_conv1(out_sr))
        out_dn = self.dn_hidden(out_dn)
        noise_residual = self.dn_conv_out(out_dn)
        
        clean_channel = out_sr - noise_residual
        return clean_channel


# ==============================================================================
# 2. Synthetic OFDM Time-Frequency Dataset Generator
# ==============================================================================
class OFDM_Dataset(Dataset):
    def __init__(self, num_samples, num_subcarriers=72, num_symbols=14, snr_db=10):
        self.num_samples = num_samples
        self.grid_shape = (num_subcarriers, num_symbols)
        self.snr_linear = 10**(snr_db / 10.0)
        
        # BUG FIX 1: Was missing list initialization (had comments instead of values)
        self.X_data = []   # Noisy LS Estimates
        self.Y_data = []   # Ground Truth CSI
        
        self._generate_data()

    def _generate_data(self):
        for _ in range(self.num_samples):
            true_h_real = np.random.randn(*self.grid_shape)
            true_h_imag = np.random.randn(*self.grid_shape)
            true_h_real = gaussian_filter(true_h_real, sigma=1.5)
            true_h_imag = gaussian_filter(true_h_imag, sigma=1.5)
            
            noise_variance = 1.0 / self.snr_linear
            noise_real = np.sqrt(noise_variance) * np.random.randn(*self.grid_shape)
            noise_imag = np.sqrt(noise_variance) * np.random.randn(*self.grid_shape)
            
            ls_h_real = true_h_real + noise_real
            ls_h_imag = true_h_imag + noise_imag
            
            X = np.stack((ls_h_real, ls_h_imag), axis=0)
            Y = np.stack((true_h_real, true_h_imag), axis=0)
            
            self.X_data.append(torch.tensor(X, dtype=torch.float32))
            self.Y_data.append(torch.tensor(Y, dtype=torch.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]


# ==============================================================================
# 3. Visualization: Loss Plot + Channel Heatmap Comparison
# ==============================================================================
def plot_results(loss_history, model, dataset, device):
    """
    Generates a figure with:
      Left:   Training loss curve over all epochs
      Right:  Heatmaps — Noisy Input / Model Output / Ground Truth / Error Map
    Saves to channelnet_results.png and displays on screen.
    """
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f1a')
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.4, hspace=0.5)

    # --- Panel 1: Loss Curve ---
    ax_loss = fig.add_subplot(gs[:, 0:2])
    ax_loss.set_facecolor('#1a1a2e')
    epochs_range = range(1, len(loss_history) + 1)
    ax_loss.plot(epochs_range, loss_history, color='#00d4ff', linewidth=2.5, label='Training MSE Loss')
    ax_loss.fill_between(epochs_range, loss_history, alpha=0.15, color='#00d4ff')
    ax_loss.set_xlabel('Epoch', color='white', fontsize=12)
    ax_loss.set_ylabel('MSE Loss', color='white', fontsize=12)
    ax_loss.set_title('Training Loss Convergence', color='white', fontsize=14, fontweight='bold')
    ax_loss.tick_params(colors='white')
    ax_loss.spines[:].set_color('#444466')
    ax_loss.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.2, color='#444466')

    # --- Panel 2: Channel Heatmaps ---
    model.eval()
    sample_x, sample_y = dataset[0]
    with torch.no_grad():
        estimated = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()

    # Show only the Real component (channel 0) for clarity
    noisy_map     = sample_x[0].numpy()
    estimated_map = estimated[0].numpy()
    truth_map     = sample_y[0].numpy()

    vmin = truth_map.min()
    vmax = truth_map.max()

    titles = ['Noisy LS Input', 'ChannelNet Output', 'Ground Truth']
    maps   = [noisy_map, estimated_map, truth_map]
    colors = ['#ff6b6b', '#00d4ff', '#51cf66']
    positions = [(0, 2), (0, 3), (1, 2)]

    for (row, col), title, data, color in zip(positions, titles, maps, colors):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#1a1a2e')
        im = ax.imshow(data, aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, color=color, fontsize=11, fontweight='bold')
        ax.set_xlabel('OFDM Symbols (Time)', color='white', fontsize=8)
        ax.set_ylabel('Subcarriers (Freq)', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.spines[:].set_color('#444466')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # 4th panel: pixel-wise absolute error map
    ax_err = fig.add_subplot(gs[1, 3])
    ax_err.set_facecolor('#1a1a2e')
    error_map = np.abs(estimated_map - truth_map)
    im_err = ax_err.imshow(error_map, aspect='auto', cmap='hot', origin='lower')
    ax_err.set_title('Absolute Error', color='#ffd43b', fontsize=11, fontweight='bold')
    ax_err.set_xlabel('OFDM Symbols (Time)', color='white', fontsize=8)
    ax_err.set_ylabel('Subcarriers (Freq)', color='white', fontsize=8)
    ax_err.tick_params(colors='white', labelsize=7)
    ax_err.spines[:].set_color('#444466')
    plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

    final_loss  = loss_history[-1]
    improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
    fig.suptitle(
        f'ChannelNet — OFDM Channel Estimation Results\n'
        f'Final MSE: {final_loss:.6f}  |  Loss Reduction: {improvement:.1f}%',
        color='white', fontsize=15, fontweight='bold', y=1.01
    )

    plt.savefig('channelnet_results.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\n📊 Results plot saved to: channelnet_results.png")
    plt.show()


# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train_model():
    BATCH_SIZE    = 32
    EPOCHS        = 50
    LEARNING_RATE = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Executing Deep Learning Pipeline on: {device}")
    
    print("🔄 Synthesizing OFDM Dataset (this takes ~1-2 min on CPU)...")
    train_dataset = OFDM_Dataset(num_samples=2500, snr_db=15)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Dataset ready: {len(train_dataset)} samples\n")
    
    model     = OFDM_ChannelNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    loss_history = []
    
    print("🚀 Training started...\n")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        for noisy_ls, true_h in train_loader:
            noisy_ls, true_h = noisy_ls.to(device), true_h.to(device)
            
            optimizer.zero_grad()
            estimated_h = model(noisy_ls)
            loss = criterion(estimated_h, true_h)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        # BUG FIX 2: param_groups is a list, must index with [0]
        current_lr = optimizer.param_groups[0]['lr']

        bar_len = 25
        filled  = int(bar_len * (epoch + 1) / EPOCHS)
        bar     = '█' * filled + '░' * (bar_len - filled)
        print(f"Epoch {epoch+1:>3}/{EPOCHS}  [{bar}]  MSE: {avg_loss:.6f}  LR: {current_lr:.6f}")
            
    print("\n✅ Training complete! Generating visualizations...")
    plot_results(loss_history, model, train_dataset, device)
    
    return model


if __name__ == "__main__":
    trained_model = train_model()