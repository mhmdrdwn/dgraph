import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops

class TGCN(torch.nn.Module):
    
    """
    Edited and adopted after pytorch geometric temporal library.
    Ref : https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H

    
class A3TGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        num_nodes: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
        ):
        
        super(A3TGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.num_nodes = num_nodes
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        #device = torch.device('mps')
        device = next(self.parameters()).device
        self.attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self.attention)
        self.Conv2Dblock =  Conv2Dblock(timepoints=100, num_nodes=self.num_nodes, node_features=self.in_channels)
        
    def forward(
        self,
        X: torch.FloatTensor,
        A: torch.FloatTensor,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        
        H_accum = 0
        probs = torch.nn.functional.softmax(self.attention, dim=-1)
        X = self.Conv2Dblock(X)
        
        # init HS
        H = None
        for period in range(self.periods):
            Xt = X[:, :, :, period]
            batch_size = Xt.shape[0]
            Xt = Xt.reshape(Xt.shape[0]*Xt.shape[1], Xt.shape[-1])
            At = A[:, :, :, period]
            At = torch.block_diag(*At)
            idx = (At > 0).nonzero().t().contiguous().long().to(X.device)
            row, col = idx
            w = At[row, col].float().to(X.device)           
            H = self._base_tgcn(Xt, idx, w, H)
            H = H.reshape(batch_size, self.num_nodes, self.out_channels)
            H_accum = H_accum + probs[period] * H 
            H = H.reshape(batch_size*self.num_nodes, self.out_channels)
        return H_accum
    
        

class Conv2Dblock(nn.Module):
    def __init__(self, timepoints, num_nodes, node_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_nodes, num_nodes, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(num_nodes, num_nodes, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(num_nodes, num_nodes, kernel_size=(8, 1), stride=(2, 1), padding=(3, 0))
        self.pooling = torch.nn.AdaptiveAvgPool2d((node_features, num_nodes))
        self.BN1 = nn.BatchNorm2d(num_nodes)
        
    def forward(self, x):
        device = x.device
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))
        x = torch.cat([x1,x3,x3],dim=2)
        x = x.permute(0, 3, 2, 1)
        x = self.pooling(x)
        x = x.permute(0, 3, 2, 1)
        x = self.BN1(x)
        return x
    
class EEGModel(nn.Module):
    def __init__(self, num_nodes, node_features, num_classes, num_windows, device):
        super(EEGModel, self).__init__()
        self.forwardA3TGCN = A3TGCN(in_channels=node_features, out_channels=32, periods=num_windows, num_nodes=num_nodes).to(device)
        self.backwardA3TGCN = A3TGCN(in_channels=node_features, out_channels=32, periods=num_windows, num_nodes=num_nodes).to(device)
        self.num_nodes = num_nodes
        self.BN = nn.BatchNorm1d(self.num_nodes)
        self.num_windows = num_windows
        self.fc2 = nn.Linear(self.num_nodes*64, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, X, A):
        X = self.dropout(X)
        HS1 = self.forwardA3TGCN(X, A)
        X_flip = torch.flip(X, dims=[1])
        A_flip = torch.flip(A, dims=[1])
        HS2 = self.backwardA3TGCN(X_flip, A_flip)
        HS = torch.cat((HS1, HS2), -1)
        HS = nn.functional.relu(HS)
        HS = self.BN(HS)
        HS = HS.reshape(HS.shape[0], self.num_nodes*64)
        out = self.fc2(HS)
        out = nn.LogSoftmax(dim=-1)(out)
        return out