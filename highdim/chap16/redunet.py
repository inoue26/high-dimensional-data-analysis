from typing import Tuple, List
import torch
from torch import Tensor
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from highdim.chap16.coding_rate import coding_rate_grads

class SimpleReduNet(torch.nn.Module):
    # Regulation:
    ## 1. not reduce input dimension
    ## 2. not consider convolution fast calculation (eg. Fast Fourier Transform)
    ## 3.
    def __init__(self, L:int, epsilon:float, a0:Tensor, a1:Tensor, gamma:Tensor, eta:Tensor, temperature:Tensor,
                 input_train:Tensor, label_train:Tensor, use_fast:bool=True):
        super(SimpleReduNet, self).__init__()
        self.L = L # num_layers := num of updates
        self.epsilon = epsilon # size of allowable error　
        self.a0 = torch.nn.parameter.Parameter(a0, requires_grad=False)
        self.a1 = torch.nn.parameter.Parameter(a1, requires_grad=False)
        self.gamma = torch.nn.parameter.Parameter(gamma, requires_grad=False)
        self.eta = torch.nn.parameter.Parameter(eta, requires_grad=False) # learning rate. step of update.
        self.temperature = torch.nn.parameter.Parameter(temperature, requires_grad=False) # temperature of Softmax.
        self.use_fast = use_fast

        # build (at once. decided manner)
        Elist, Cslist, input_train = self.build(input_train, label_train)
        self.Elist = torch.nn.ParameterList(Elist)
        self.Cslist = torch.nn.ParameterList(Cslist)

    @torch.no_grad()
    def build(self, input_train:Tensor, label_train:Tensor)->Tuple[Tensor, ...]:
        # input_train: all training samples. space_dim x n_samples or space_dim x space_dim x n_samples
        # label_train: n_classes x n_samples
        Elist = []
        Cslist = []
        for l in range(self.L):
            grads, E, Cs = coding_rate_grads(input_train, label_train, self.a0, self.a1,
                                             self.gamma, use_fast=self.use_fast, return_params=True)
            Elist.append(E)
            Cslist.append(Cs)
            input_train = input_train + self.eta*grads
            # normalize
            input_train = input_train / torch.norm(input_train, p='fro', dim=0, keepdim=True)
        return Elist, Cslist, input_train

    def predict(self, Z:Tensor, P:Tensor|None=None, L:int|None=None):
        # Z: space_dim x n_samples
        with torch.no_grad():
            return self.forward(Z, P, L)

    def forward(self, Z:Tensor, P:Tensor|None, L:int|None):
        # Z: space_dim x n_samples
        if L is None:
            L = self.L
        for l in range(L):
            if P is None:
                sims = -torch.norm(self.Cslist[l] @ Z, p='fro', dim=1) # num_classes x space_dim x n_samples -> num_classes x n_samples
                P_hat = torch.nn.functional.softmax(sims/self.temperature, dim=0) # num_classes x n_samples
            else:
                P_hat = P
            grads = self.Elist[l] @ Z - torch.einsum('c,cin->in', self.gamma,
                                                     torch.einsum('cij,cjn->cin', self.Cslist[l],(P_hat.unsqueeze(1)*Z))
                                                     )
            Z = Z + self.eta * grads
            Z = Z / torch.norm(Z, p='fro', dim=0, keepdim=True)
        return Z




if __name__ == '__main__':
    import time
    SPACE_DIM = 4
    N_SAMPLES = 100
    N_CLASSES = 3
    Z = torch.rand([SPACE_DIM, N_SAMPLES]) # space_dim x n_samples
    P = torch.rand([N_CLASSES, N_SAMPLES]) # n_classes x n_samples
    P = P / P.sum(0) # to prob
    # a0 = torch.rand([1]) # float
    # a1 = torch.rand([N_CLASSES]) # num_classes
    # gamma = torch.rand([N_CLASSES])
    a0 = torch.ones(1) # float 1だとclassが潰れる
    a1 = torch.ones([N_CLASSES]) # num_classes
    gamma = torch.ones([N_CLASSES])

    epsilon = torch.tensor([0.1])
    eta = torch.tensor([0.5])
    temperature = torch.tensor([1.0]) * 0.01
    L = 100

    model = SimpleReduNet(L, epsilon, a0, a1, gamma, eta, temperature, Z, P, use_fast=True)

    finZ = model.predict(Z, None, None)
    print(finZ[:, :5])
    finZ = model.predict(Z, P, None)
    print(finZ[:, :5])