import matplotlib.pyplot as plt
import torch
from transformer import *
import seaborn as sns


def test_attention():
    seq_len, d_k = 3, 2
    pl.seed_everything(42)
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    values, attention = scaled_dot_product(q, k, v)
    print(f"Q:{q}\n, K:{k}\n,V:{v}\n, Values:{values}\n,Attention:{attention}\n")


def test_multi_head_atte():
    input_dim = 512
    embed_dim = 64
    multi_head_atte = MultiHeadAttention(input_dim, embed_dim, num_heads=8)
    x = torch.randn(2, 120, 512)
    value = multi_head_atte(x)
    pass


def test_position_encode():
    encod_block = PositionalEncoding(d_model=48, max_len=96)
    pe = encod_block.pe.squeeze().T.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
    ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
    plt.show()

    sns.set_theme()
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))
    ax = [a for a_list in ax for a in a_list]
    for i in range(len(ax)):
        ax[i].plot(np.arange(1, 17), pe[i, :16], color=f'C{i}', marker="o", markersize=6, markeredgecolor="black")
        ax[i].set_title(f"Encoding in hidden dimension {i + 1}")
        ax[i].set_xlabel("Position in sequence", fontsize=10)
        ax[i].set_ylabel("Positional encoding", fontsize=10)
        ax[i].set_xticks(np.arange(1, 17))
        ax[i].tick_params(axis='both', which='major', labelsize=10)
        ax[i].tick_params(axis='both', which='minor', labelsize=8)
        ax[i].set_ylim(-1.2, 1.2)
    fig.subplots_adjust(hspace=0.8)
    sns.reset_orig()
    plt.show()


def test_scheduler():
    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-3)
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)

    # Plotting
    epochs = list(range(2000))
    sns.set()
    plt.figure(figsize=(8, 3))
    plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title("Cosine Warm-up Learning Rate Scheduler")
    plt.show()
    sns.reset_orig()


if __name__ == "__main__":
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "../data"

    # Setting the seed
    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    # test_attention()
    # test_multi_head_atte()
    # test_position_encode()
    test_scheduler()
    pass
