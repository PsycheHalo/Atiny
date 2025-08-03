from typing import List, Optional, Union, Tuple
import math
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, ParamsT, _use_grad_for_differentiable,_get_scalar_dtype)

__all__ = ['Atiny',]

class Atiny(Optimizer):
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, Tensor] = 1e-3,
                 smooth: Union[float, Tensor] = 10,
                 weight_decay:  Union[float, Tensor] = 0,
                 ldr: Union[float, Tensor] = 10,
                 autoIndividuateLR=False, #自动个性化学习率,目前的版本是根据维度尺寸自动计算.
                 ):
        
        if not lr>0:
            raise ValueError(f'Invalid 学习速率: {lr}')
        if not smooth>0: 
            raise ValueError(f'Invalid 平滑程度: {smooth}')
        if not weight_decay>=0:
            raise ValueError(f'Invalid 权重衰减: {weight_decay}')
        if not ldr>=0:
            raise ValueError(f'Invalid 学习衰减率: {lr}')
        
        defaults = dict(lr=lr,smooth=smooth,weight_decay=weight_decay,ldr=ldr,autoIndividuateLR=autoIndividuateLR)
        super().__init__(params, defaults)

    def _norm(self,x):
        return torch.linalg.norm(x,keepdim=False)
    
    def _concord(self,x1,x2):
        concord=torch.nn.functional.cosine_similarity(x1.view(-1),x2.view(-1),dim=0,eps=0)
        concord=(concord.asin()*2/math.pi+1).clamp(max=1,min=0)
        concord[~torch.isfinite(concord)]=1
        
        return concord

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state['warm']):
                    warm_val = float(p_state["warm"])
                    p_state["warm"] = torch.tensor(warm_val, dtype=_get_scalar_dtype())

    #@_use_grad_for_differentiable
    def step(self, closure=None, loss=None):
        self._cuda_graph_capture_health_check()

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss is None:
            loss=torch.tensor(1, dtype=_get_scalar_dtype())
        
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if not torch.isfinite(p.data).all():
                        print("tensor.data is not finite  --Optimizer")
                    p.data[~torch.isfinite(p.data)]=0
                    if p.requires_grad==False:
                        continue
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        print("tensor.grad is not finite  --Optimizer")
                    p.grad.data[~torch.isfinite(p.grad.data)]=0
                    if p.grad.data.is_sparse:
                        raise RuntimeError('AtinyOptimizer does not support sparse gradients')
                    
                    state = self.state[p]  #get state dict for this param
                    if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                        state['warm'] = torch.tensor(0, dtype=_get_scalar_dtype())
                        state['flat'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['dNormAccumulat'] = torch.tensor(0, dtype=_get_scalar_dtype(),device=p.device)
                        
                    smooth=group['smooth']

                    state['warm']=(state['warm']*smooth+1)/(smooth+1)
                    
                    k=(state['flat']+p.grad).abs_()
                    k/=state['flat'].abs().add_(p.grad.abs())
                    if group['ldr']>0:
                        k/=(1+state['dNormAccumulat'])
                    k[~torch.isfinite(k)]=0
                    k.clamp_(min=0,max=1)
                    
                    state['flat']=((state['flat']*smooth).add_(p.grad)).div_(smooth+1)
                    state['flat'][~torch.isfinite(state['flat'])]=0
                    
                    if group['autoIndividuateLR']:
                        n0Dim=0
                        for d in p.size():
                            if d>1:
                                n0Dim+=1;
                        n0Dim=min(n0Dim,2)
                        deDim=1/(p.numel()**(0.5**n0Dim))
                    else:
                        deDim=1
                    
                    d=(state['flat'].sign().mul_(k)).mul_(group['lr']*state['warm']*deDim)
                    d[~torch.isfinite(d)]=0
                    dNorm=self._norm(d)
                    dNorm[~torch.isfinite(dNorm)]=0
                    
                    p.data-=d
                    
                    if group['weight_decay']>0:
                        wd=group['weight_decay']*deDim
                        norm=self._norm(p.data)*wd
                        wdk=norm.asinh()/norm
                        wdk[~torch.isfinite(wdk)]=1
                        wdk.clamp_(min=0,max=1)
                        p.data*=wdk
                        
                    p.data[~torch.isfinite(p.data)]=0
                    
                    if group['ldr']>0:
                        state['dNormAccumulat']+=group['ldr']*dNorm*deDim
                    
                    
        return loss


