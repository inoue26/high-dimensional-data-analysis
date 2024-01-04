from typing import Tuple, List
import torch
from torch import Tensor
import numpy as np


def coding_rate(Z:Tensor, P:Tensor, a0:float, a1:Tensor, gamma:Tensor, use_fast:bool=False)->Tensor:
    # Z: space_dim x n_samples or space_dim x space_dim x n_samples
    # P: n_classes x n_samples
    # a0: (float)
    # a1: num_classes
    # gamma: num_classes
    space_dim = len(Z)
    term1 = 1/2 * torch.logdet(torch.eye(space_dim) + a0*Z@Z.T) # cov_mat = Z@Z.T
    if use_fast:
        a1 = a1.unsqueeze(-1).unsqueeze(-1)
        term2 = 1/2 * torch.sum(gamma @ torch.logdet(torch.eye(space_dim) +  a1*(P.unsqueeze(1)*Z)@Z.T))
    else:
        # original formula
        term2 = 0
        for k, p in enumerate(P):
            # p: n_samples
            term2_class = 1/2*gamma[k]*torch.logdet(torch.eye(space_dim) + a1[k]*Z@torch.diag(p)@Z.T)
            term2 += term2_class

    return term1 - term2  # 0 dim

def coding_rate_grads_term1(Z:Tensor, P:Tensor, a0:float, use_fast:bool=False,
                            returnE:bool=False)->Tensor:
    space_dim = len(Z)
    if use_fast: # TODO: but no idea.
        # E
        E = a0*torch.linalg.inv(torch.eye(space_dim)+ a0*Z@Z.T) # space_dim x space_dim
        term1 = E @ Z # space_dim x n_samples
    else:
        # E
        E = a0*torch.linalg.inv(torch.eye(space_dim)+ a0*Z@Z.T) # space_dim x space_dim
        term1 = E @ Z # space_dim x n_samples
    if returnE:
        return term1, E
    else:
        return term1, None

def coding_rate_grads_term2(Z:Tensor, P:Tensor, a1:Tensor, gamma:Tensor, use_fast:bool=False,
                            returnC:bool=False)->Tensor:
    space_dim = len(Z)
    if use_fast:
        a1 = a1.unsqueeze(-1).unsqueeze(-1)
        Cs = a1*torch.linalg.inv(torch.eye(space_dim) + a1*(P.unsqueeze(1)*Z)@Z.T) # num_class x space_dim x space_dim
        # (num_class)@(num_class x space_dim x space_dim)@(num_classes x space_dim x num_samples := (num_classes x num_samples x 1)*(space_dim x num_samples))
        term2 = torch.einsum('c,cin->in', gamma, torch.einsum('cij,cjn->cin', Cs,(P.unsqueeze(1)*Z)))
    else:
        # C
        Cs = []
        term2 = 0
        for k, p in enumerate(P):
            C = a1[k]*torch.linalg.inv(torch.eye(space_dim) + a1[k]*Z@torch.diag(p)@Z.T) # space_dim x space_dim
            term2 += gamma[k]*C@Z@torch.diag(p) # space_dim x n_samples
            if returnC:
                Cs.append(C)
        if returnC:
            Cs=torch.stack(Cs, dim=0)
    if returnC:
        return term2, Cs
    else:
        return term2, None

def coding_rate_grads(Z:Tensor, P:Tensor, a0:float, a1:Tensor, gamma:Tensor, use_fast:bool=False,
                      return_params:bool=False)->Tuple[Tensor, Tensor | None, Tensor | None]:
    # Args:
    ## Z: space_dim x n_samples or space_dim x space_dim x n_samples
    ## P: n_classes x n_samples
    ## a0: (float)
    ## a1: num_classes
    ## gamma: num_classes
    # Returns:
    ## E: space_dim x space_dim
    ## Cs: num_classes x space_dim x space_dim
    term1, E = coding_rate_grads_term1(Z, P, a0, use_fast=use_fast, returnE=return_params)
    term2, Cs = coding_rate_grads_term2(Z, P, a1, gamma, use_fast=use_fast, returnC=return_params)

    return term1 - term2, E, Cs



if __name__ == '__main__':
    import time
    SPACE_DIM = 4
    N_SAMPLES = 5
    N_CLASSES = 2
    Z = torch.rand([SPACE_DIM, N_SAMPLES]) # space_dim x n_samples
    P = torch.rand([N_CLASSES, N_SAMPLES]) # n_classes x n_samples
    P = P / P.sum(0) # to prob
    a0 = torch.rand([1]) # float
    a1 = torch.rand([N_CLASSES]) # num_classes
    gamma = torch.rand([N_CLASSES])

    ###########################
    ### TEST1: coding rate test
    ###########################
    cr1 = coding_rate(Z, P, a0, a1, gamma, use_fast=False)
    cr2 = coding_rate(Z, P, a0, a1, gamma, use_fast=True)
    print('coding rate: ', cr1, cr2)
    assert np.isclose(cr1, cr2, atol=1e-5), 'cr1 and cr2 should have the same value. but have ({}, {})'.format(cr1, cr2)
    start = time.time()
    for _ in range(100):
        coding_rate(Z, P, a0, a1, gamma, use_fast=False)
    end1 = time.time()
    for _ in range(100):
        coding_rate(Z, P, a0, a1, gamma, use_fast=True)
    end2 = time.time()
    print('duration[sec]: cr1 {}, cr2 {}'.format(end1-start, end2-end1))
    assert end1-start  > end2-end1, 'latter implementation should be faster than former.'
    ###########################
    ### TEST1: coding rate gradient test
    ###########################
    grad_cr1, _, _ = coding_rate_grads(Z, P, a0, a1, gamma, use_fast=False)
    grad_cr2, _, _ = coding_rate_grads(Z, P, a0, a1, gamma, use_fast=True)
    print('gradient of coding rate: ', grad_cr1, grad_cr2)
    assert np.isclose(grad_cr1, grad_cr2, atol=1e-5).all(), 'grad_cr1 and grad_cr2 should have the same value. but have ({}, {})'.format(grad_cr1, grad_cr2)
    start = time.time()
    for _ in range(100):
        coding_rate_grads(Z, P, a0, a1, gamma, use_fast=False)
    end1 = time.time()
    for _ in range(100):
        coding_rate_grads(Z, P, a0, a1, gamma, use_fast=True)
    end2 = time.time()
    print('duration[sec]: grad_cr1 {}, grad_cr2 {}'.format(end1-start, end2-end1))
    assert end1-start  > end2-end1, 'latter implementation should be faster than former.'