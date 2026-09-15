import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_scatter import scatter_mean, scatter_add




class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, z_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x, batch):
        
        g = global_add_pool(x, batch)
        h = self.mlp(g)
        mu = self.mu(h)
        logvar = self.logvar(h)
        sigma = torch.exp(0.5 * logvar).clamp_min(1e-6)
        # reparameterization
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z, mu, sigma, g   



class ReferenceDistributions(nn.Module):
    def __init__(self, dim, num_bases=16):
        super().__init__()
        self.num_bases = num_bases
        self.mu_bases = nn.Parameter(torch.empty(num_bases, dim))
        nn.init.uniform_(self.mu_bases, a=-1.0, b=1.0)
        self.log_sigma_bases = nn.Parameter(torch.zeros(num_bases, dim))
        self.weight_history = []
    def forward(self, mu_t, sigma_t):
        sigma_b = torch.exp(self.log_sigma_bases) + 1e-6
        mu_diff = mu_t.unsqueeze(1) - self.mu_bases.unsqueeze(0)
        sigma_diff = sigma_t.unsqueeze(1) - sigma_b.unsqueeze(0)
        dist2 = (mu_diff**2 + sigma_diff**2).sum(dim=-1)

        weights = F.softmax(-dist2 / 0.15, dim=-1)  #tau =0.15
        
        mu_proj = torch.einsum("bk,kd->bd", weights, self.mu_bases)
        sigma_proj = torch.einsum("bk,kd->bd", weights, sigma_b)
        sigma_proj = sigma_proj.clamp(min=1e-3, max = 2.0)

        return mu_proj, sigma_proj, weights, dist2






class FeatureTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, dropout):
        super(FeatureTransformMLP, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features):
        x = self.mlp(node_features)
        return self.dropout_layer(x)


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super().__init__()
        self.residuals = residuals
        self.dropout_layer = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*n_node_f + n_edge_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.dropout_layer(out)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.heads = 4
        self.conv = GATv2Conv(n_node_f, int(out_dim / self.heads), edge_dim=n_edge_f, heads=self.heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = F.relu(self.conv(x, edge_index, edge_attr))
        if self.residuals:
            out = out + x
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_f, glob_f_in, glob_f_hidden, glob_f_out, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_node_f + glob_f_in, glob_f_hidden),
            nn.ReLU(),
            nn.Linear(glob_f_hidden, glob_f_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, global_add_pool(x, batch=batch)], dim=1)
        out = self.dropout_layer(out)
        return self.global_mlp(out)



def WBRO(x, graph_batch, style_bases):
    batch = graph_batch.batch
    one = torch.ones_like(batch, dtype=x.dtype, device = x.device)
    counts = scatter_add(one, batch, dim=0)
    counts = counts.clamp(min=1).unsqueeze(1)
    mu_t = scatter_mean(x, batch, dim=0)
    diff = x - mu_t[batch]
    var_t = scatter_add(diff*diff, batch, dim=0)
    var_t = var_t / counts
    sigma_t = torch.sqrt(var_t + 1e-6)
    sigma_t = sigma_t.clamp(min=1e-3)
    mu_proj, sigma_proj, weights, diff2 = style_bases(mu_t, sigma_t)
    x_norm = (x - mu_t[batch]) / sigma_t[batch]
    x_shifted = x_norm * sigma_proj[batch] + mu_proj[batch]
    return x_shifted, mu_t, sigma_t, mu_proj, sigma_proj, weights, diff2




class WLRL18(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(WLRL18, self).__init__()

        self.NodeTransform = FeatureTransformMLP(in_channels, 256, 64, dropout=dropout_prob)

        self.layer1 = self.build_layer(node_f=64, node_f_hidden=64, node_f_out=64,
                                       edge_f=edge_dim, edge_f_hidden=64, edge_f_out=64,
                                       glob_f=384, glob_f_hidden=384, glob_f_out=384,
                                       residuals=False, dropout=conv_dropout_prob
                                       )

        self.node_bn1 = BatchNorm1d(64)
        self.edge_bn1 = BatchNorm1d(64)
        self.u_bn1 = BatchNorm1d(384)

        self.layer2 = self.build_layer(node_f=64, node_f_hidden=64, node_f_out=64,
                                       edge_f=64, edge_f_hidden=64, edge_f_out=64,
                                       glob_f=384, glob_f_hidden=384, glob_f_out=384,
                                       residuals=False, dropout=conv_dropout_prob
                                       )

        self.ReferenceDistributions = ReferenceDistributions(dim=64, num_bases=128)  #1024
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self,
                    node_f, node_f_hidden, node_f_out,
                    edge_f, edge_f_hidden, edge_f_out,
                    glob_f, glob_f_hidden, glob_f_out,
                    residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, glob_f, glob_f_hidden, glob_f_out, dropout=dropout)
        )

    def forward(self, graphbatch):

        edge_index = graphbatch.edge_index
        x = self.NodeTransform(graphbatch.x)
        x, edge_attr, u = self.layer1(x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)
        x, mu_t, sigma_t, mu_proj, sigma_proj, weights, dist2  = WBRO(x, graphbatch, self.ReferenceDistributions)
        _, _, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)

        # u:[512, 384], x:[28938,64]
        u = self.dropout_layer(u)
        # Fully-Connected Layers
        out = self.fc1(u)
        out = F.relu(out)
        out = self.fc2(out)

        extra = {
            "weights": weights,
            "dist2": dist2,
            "mu_t": mu_t, "sigma_t": sigma_t,
            "mu_proj": mu_proj, "sigma_proj": sigma_proj
        }
        return out, extra






