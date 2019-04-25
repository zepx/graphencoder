#python3

import numpy as np
import argparse
from tqdm import tqdm
from sklearn.datasets import load_wine
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch import nn, optim
import torch

from model import GraphEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine', help='Dataset to use')
parser.add_argument('-l', '--layers', nargs='+', type=int, default=[128, 64, 128], help='Sparsity Penalty Parameter')
parser.add_argument('-b', '--beta', type=float, default=0.01, help='Sparsity Penalty Parameter')
parser.add_argument('-p', '--rho', type=float, default=0.5, help='Prior rho')
parser.add_argument('-lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('-epoch', type=int, default=200, help='Number of Training Epochs')
parser.add_argument('-device', type=str, default='gpu', help='Train on GPU or CPU')

args = parser.parse_args()
device = torch.device('cuda' if args.device == 'gpu' else 'cpu')

def main():
    if args.dataset.lower() == 'wine':
        data = load_wine()
    else:
        raise Exception('Invalid dataset specified')

    X = data.data
    Y = data.target
    k = len(np.unique(Y))

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    
    # Obtain Similarity matrix
    S = cosine_similarity(X, X)

    D = np.diag(1.0 / np.sqrt(S.sum(axis=1)))
    X_train = torch.tensor(D.dot(S).dot(D)).float().to(device)

    layers = [len(X_train)] + args.layers + [len(X_train)]

    model = GraphEncoder(layers, k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    with tqdm(total=args.epoch) as tq:
        for epoch in range(1, args.epoch + 1):
            optimizer.zero_grad()
            X_hat = model(X_train)
            loss = model.loss(X_hat, X_train, args.beta, args.rho)
            nmi = normalized_mutual_info_score(model.get_cluster(), Y, average_method='arithmetic')

            loss.backward()
            optimizer.step()

            tq.set_postfix(loss='{:.3f}'.format(loss), nmi='{:.3f}'.format(nmi))
            tq.update()
        print(model.get_cluster())


if __name__ == '__main__':
    main()
