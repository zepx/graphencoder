#python3

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.cluster import KMeans

class GraphEncoder(nn.Module):
    def __init__(self, layers, clusters):
        super(GraphEncoder, self).__init__()

        self.layers = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(layers[0], layers[1]),
            'sig1': nn.Sigmoid(),
            'lin2': nn.Linear(layers[1], layers[2]),
            'sig2': nn.Sigmoid(),
            'lin3': nn.Linear(layers[2], layers[3]),
            'sig3': nn.Sigmoid(),
            'lin4': nn.Linear(layers[3], layers[4]),
            'sig4': nn.Sigmoid(),
            }))
        self.clusters = clusters

        self.outputs = {}

        self.layers[0].register_forward_hook(self.get_activation('lin1'))
        self.layers[2].register_forward_hook(self.get_activation('lin2'))
        self.layers[4].register_forward_hook(self.get_activation('lin3'))
    
    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def forward(self, x):
        output = self.layers(x)
        return output

    def layer_activations(self, layername):
        return torch.mean(torch.sigmoid(self.outputs[layername]), dim=0)

    def sparse_result(self, rho, layername):
        rho_hat = self.layer_activations(layername)
        return rho * np.log(rho) - rho * torch.log(rho_hat) + (1 - rho) * np.log(1 - rho) \
                - (1 - rho) * torch.log(1 - rho_hat)

    def kl_div(self, rho):
        first = torch.mean(self.sparse_result(rho, 'lin1'))
        second = torch.mean(self.sparse_result(rho, 'lin2'))
        return first + second

    def get_index_by_name(self, name):
        return list(dict(self.layers.named_children()).keys()).index(name)

    def loss(self, x_hat, x, beta, rho):
        loss = F.mse_loss(x_hat, x) + beta * self.kl_div(rho)
        return loss

    def get_cluster(self):
        kmeans = KMeans(n_clusters=self.clusters).fit(self.outputs['lin2'].detach().cpu().numpy())
        self.centroids = kmeans.cluster_centers_
        return kmeans.labels_
