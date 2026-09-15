import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from Dataset import *
from torch_geometric.loader import DataLoader
from model import *
from sklearn.linear_model import LinearRegression

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))


def load_model_state(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()  # Set the model to evaluation mode
    return model


def compute_mae(y_true, y_pred):
    """
    y_true, y_pred: 1-D numpy arrays (unscaled)
    returns: float
    """
    return np.mean(np.abs(y_true - y_pred))

# helper: SD as std of residuals after fitting linear regression y_true ~ y_pred
def compute_sd(y_true, y_pred):
    """
    Fit LinearRegression(y_pred -> y_true) and compute sample std of residuals:
    sd = sqrt( sum((y_true - y_fit)^2) / (n - 1) )
    y_true, y_pred: 1-D numpy arrays
    returns: float
    """
    y = y_true.reshape(-1, 1)
    f = y_pred.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(f, y)
    y_fit = lr.predict(f)
    n = len(y_true)
    if n <= 1:
        return 0.0
    resid2_sum = ((y - y_fit) ** 2).sum()
    sd = np.sqrt(resid2_sum / (n - 1))
    # return scalar
    return float(sd)


import time
# Evaluation Function
#-------------------------------------------------------------------------------------------------------------------------------
def evaluate(models, loader, criterion, device):
    
    # Initialize variables to accumulate the evaluation results
    total_loss = 0.0
    y_true = []
    y_pred = []
    id = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        t = []
        for graphbatch in loader:

            graphbatch.to(device)
            targets = graphbatch.y

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # Forward pass EMSEMBLE MODEL
            outputs = []
            for i in range(len(models)):
                # print(i)
                model = models[i]
                outputs.append(model(graphbatch)[0].view(-1))
                # outputs.append(model(graphbatch).view(-1))


            output = torch.mean(torch.stack(outputs), dim=0)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(t1, t2, t2-t1)
            t.append(t2-t1)




            loss = criterion(output, targets)
            # Accumulate loss and collect the true and predicted values for later use
            total_loss += loss.item()
            y_true.extend(targets.tolist())
            y_pred.extend(output.tolist())
            id.extend(graphbatch.id)
        mean = np.mean(t)
        std = np.std(t)
        print('mean and std', mean, std)

    # Calculate evaluation metrics
    eval_loss = total_loss / len(loader)
    # Pearson Correlation Coefficient
    corr_matrix = np.corrcoef(y_true, y_pred)
    r = corr_matrix[0, 1]
    # Link the predictions to the corresponding ids in a dictionary
    id_to_pred = dict(zip(id, zip(y_true, y_pred)))
    # R2 Score
    r2_score = 1 - np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / np.sum((np.array(y_true) - np.mean(np.array(y_true))) ** 2)

    # RMSE in pK unit
    min=0
    max=16
    true_labels_unscaled = torch.tensor(y_true) * (max - min) + min
    predictions_unscaled = torch.tensor(y_pred) * (max - min) + min
    rmse = criterion(predictions_unscaled, true_labels_unscaled)



    true_unscaled = true_labels_unscaled.numpy()
    pred_unscaled = predictions_unscaled.numpy()
    # MAE (in pK unit)
    mae_val = compute_mae(true_unscaled, pred_unscaled)

    # SD (std of residuals after linear fit, in pK unit)
    sd_val = compute_sd(true_unscaled, pred_unscaled)

    return eval_loss, r, rmse, r2_score, mae_val, sd_val, true_labels_unscaled, predictions_unscaled, id_to_pred
#-------------------------------------------------------------------------------------------------------------------------------


def plot_coupling_heatmap(pi, title, save_path=None):
    """
    pi: numpy array, shape (B, K)
    title: figure title
    """

    
    pi = pi.cpu()
    pi = pi / (pi.sum(axis=1, keepdims=True) + 1e-12)

    row_order = np.argsort(np.argmax(pi, axis=1))
    pi_sorted = pi[row_order]

    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        pi_sorted,
        cmap="viridis",
        xticklabels=[f"Anchor {i + 1}" for i in range(pi.shape[1])],
        yticklabels=False,
        cbar=True
    )
    plt.title(title, fontsize=12)
    plt.xlabel("Reference Gaussian Components (K)")
    plt.ylabel("Complexes in Batch (sorted)")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



# Plotting Functions
#-------------------------------------------------------------------------------------------------------------------------
def plot_error_histogram(ax, errors, title):
    n, bins, patches = ax.hist(errors, bins=50, color='blue', edgecolor='black')
    
    # Add text on top of each column
    for count, patch in zip(n, patches):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f'{int(count)}', 
                ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel('Absolute Error (pK)')
    ax.set_ylabel('Frequency')



def plot_predictions_show_in_paper(y_true, y_pred, title, metrics='', filepath=None, axislim=12):
    """Plot one publication-sized predicted-versus-experimental panel.

    ``title`` is retained for compatibility with existing calls. Panel titles
    are added in LaTeX so that paired panels use one consistent typography.
    """

    # Match the typography and palette used by the manuscript's other figures.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}."
        )
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must contain only finite values.")

    color_points = "#7896B8"
    color_identity = "#5A5A5A"
    color_edge = "#4D4D4D"
    color_grid = "#D9D9D9"

    # Generate the panel near its final double-column display size.
    fig, ax = plt.subplots(figsize=(3.20, 3.05))
    ax.scatter(
        y_true,
        y_pred,
        s=15,
        alpha=0.78,
        color=color_points,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    ax.plot(
        [0, axislim],
        [0, axislim],
        color=color_identity,
        linewidth=0.75,
        alpha=0.85,
        linestyle=(0, (4, 2)),
        zorder=2,
    )

    if metrics:
        ax.text(
            0.055,
            0.955,
            metrics,
            fontsize=7.0,
            color=color_edge,
            linespacing=1.20,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=4,
        )

    ax.set_xlabel(r"Experimental $pK$", labelpad=2.5)
    ax.set_ylabel(r"Predicted $pK$", labelpad=2.5)
    ax.set_xlim(0, axislim)
    ax.set_ylim(0, axislim)
    tick_step = 2 if axislim >= 8 else 1
    ticks = np.arange(0, axislim + 0.5 * tick_step, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", length=2.5, width=0.65, pad=1.5, colors=color_edge)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=color_grid, linewidth=0.35, alpha=0.35)
    ax.spines["left"].set_color(color_edge)
    ax.spines["bottom"].set_color(color_edge)

    fig.tight_layout(pad=0.65)
    if filepath is not None:
        print(f"Using manuscript scatter style v2; saving to: {filepath}")
        output_stem = os.path.splitext(filepath)[0]
        output_format = os.path.splitext(filepath)[1].lower().lstrip(".") or "pdf"
        fig.savefig(
            filepath,
            format=output_format,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        # Companion files support editable assembly and journal submission.
        fig.savefig(
            f"{output_stem}.svg",
            format="svg",
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            f"{output_stem}.tiff",
            format="tiff",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
#-------------------------------------------------------------------------------------------------------------------------



def parse_args():

    parser = argparse.ArgumentParser(description="Testing Parameters and Input Dataset Control")

    # REQUIRED Arguments
    parser.add_argument("--stdicts", type=str, default='./B6AEPL_train_cleansplit_WLRL18_seed0_tau_0.15_K_128_early_stop_20260818/None_f0_best_stdict.pt,./B6AEPL_train_cleansplit_WLRL18_seed0_tau_0.15_K_128_early_stop_20260818/None_f1_best_stdict.pt,./B6AEPL_train_cleansplit_WLRL18_seed0_tau_0.15_K_128_early_stop_20260818/None_f2_best_stdict.pt,./B6AEPL_train_cleansplit_WLRL18_seed0_tau_0.15_K_128_early_stop_20260818/None_f3_best_stdict.pt,./B6AEPL_train_cleansplit_WLRL18_seed0_tau_0.15_K_128_early_stop_20260818/None_f4_best_stdict.pt', help="String of comma-separated paths to stdicts that should be tested as an ensemble")
    parser.add_argument("--dataset_path", default='/mnt/CuicanYu_Li2/GEMS_data/GEMS_pytorch_datasets/B6AEPL_casf2016_indep.pt', help="The path to the test dataset pt file")

    # OPTIONAL Arguments
    parser.add_argument("--model_arch", default="WLRL18", help="The name of the model architecture")
    parser.add_argument("--save_path", default="./B6AEPL_train_cleansplit_WLRL18_seed0_tau_0.15_K_128_early_stop_20260818", help="The path where the results should be exported to")
    return parser.parse_args()

args = parse_args()


# Paths
dataset_path = args.dataset_path
stdicts = args.stdicts.split(',')
save_path = args.save_path

if save_path == None: save_path = os.path.dirname(dataset_path)

# Load the datasets
test_dataset = torch.load(dataset_path)
test_loader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)
print(f'Dataset: {dataset_path}')

# Emsemble Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_arch = args.model_arch
conv_dropout_prob = 0
dropout_prob = 0
criterion = RMSELoss()

node_feat_dim = test_dataset[0].x.shape[1]
edge_feat_dim = test_dataset[0].edge_attr.shape[1]

model_class = getattr(sys.modules[__name__], model_arch)
models = [model_class(
            dropout_prob=dropout_prob, 
            in_channels=node_feat_dim,
            edge_dim=edge_feat_dim,
            conv_dropout_prob=conv_dropout_prob).float().to(device)
            for _ in range(len(stdicts))]


## MODEL NAME ##
model_paths = list(stdicts)
#for m in model_paths: print(m)
models = [load_model_state(model, path) for model, path in zip(models, model_paths)]
print('Loaded models:')
print(model_paths)



# Run inference
loss, r, rmse, r2_score, mae, sd, y_true, y_pred, id_to_pred = evaluate(models, test_loader, criterion, device)
print('RMSE',rmse)
print('R', r)
print('MAE',mae)
print('SD', sd)

# Plotting
#-------------------------------------------------------------------------------------------------------------------------
test_dataset_name = os.path.basename(dataset_path).split('.')[0]

# Save the predictions to a json file
with open(os.path.join(save_path, f'{test_dataset_name}_predictions.json'), 'w', encoding='utf-8') as json_file:
    json.dump(id_to_pred, json_file, ensure_ascii=False, indent=4)

# Save Predictions Scatterplot
# filepath = os.path.join(save_path, f'{test_dataset_name}_predictions.png')
# plot_predictions(y_true, y_pred, test_dataset_name, metrics=f"R = {r:.3f}\nRMSE = {rmse:.3f}", filepath=filepath, axislim=14)
filepath = os.path.join(save_path, f'{test_dataset_name}_predictions.pdf')
plot_predictions_show_in_paper(y_true, y_pred, test_dataset_name, metrics=f"$R$ = {r:.3f}\nRMSE = {rmse:.3f}", filepath=filepath, axislim=12)

print(f'Predictions saved to {os.path.join(save_path, f"{test_dataset_name}_predictions")}')
#-------------------------------------------------------------------------------------------------------------------------
