'''
Taken from: 

https://github.com/zihangm/riemannian-streaming-cca


'''


import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

batch_size = 100 # the block size for streaming PCA part
class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class cca_loss(torch.nn.Module):
    def __init__(self, outdim_size, use_all_singular_values, device):
        super(cca_loss, self).__init__()
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        self.d = outdim_size
        d = self.d
        self.k = 10
        self.U = Variable(torch.rand([d, self.k])).cuda()
        self.U = self.U / torch.norm(self.U, dim=0, keepdim = True)
        self.V = Variable(torch.rand([d, self.k])).cuda()
        self.V = self.V / torch.norm(self.V, dim=0, keepdim = True)
        self.Qx, _ = torch.qr(torch.rand(d, self.k), True)
        self.Qy, _ = torch.qr(torch.rand(d, self.k), True)
        self.Qx = self.Qx.cuda()
        self.Qy = self.Qy.cuda()
        self.Tx = torch.eye(self.k).cuda()
        self.Ty = torch.eye(self.k).cuda()

        self.clock = 0

    def retr(self, X, direction, step_size):
        U,S,V = (X + direction * step_size).svd()
        return torch.matmul(U, V.transpose(0,1))

    def invretr(self, X, Y):
        return torch.matmul(Y, torch.inverse(torch.matmul(X.transpose(0,1), Y))) - X

    def invretr_batch(self, X, Y):
        return torch.matmul(Y, torch.inverse(torch.matmul(X.transpose(1,2), Y))) - X

    def retrqr(self, X, direction):
        Q,R = torch.qr(X + direction,True)
        D = torch.diag(torch.sign(torch.diag(R)+0.5))
        return Q.mm(D)

    def loss(self, H1, H2, mode):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """
        d = self.d
        # normalize the input
        H1 = (H1 - H1.mean(dim=0).unsqueeze(dim=0) )
        H2 = (H2 - H2.mean(dim=0).unsqueeze(dim=0) )

        self.U = self.U.detach()
        self.V = self.V.detach()
        self.Qx = self.Qx.detach()
        self.Qy = self.Qy.detach()
        self.Tx = self.Tx.detach()
        self.Ty = self.Ty.detach()

        #'''
        alpha = 1
        if self.clock % 1 ==0:
                self.U_rnd = Variable(torch.rand([d, self.k])).cuda()
                self.U_rnd = self.U_rnd / torch.norm(self.U_rnd, dim=0, keepdim = True)
                self.V_rnd = Variable(torch.rand([d, self.k])).cuda()
                self.V_rnd = self.V_rnd / torch.norm(self.V_rnd, dim=0, keepdim = True)
                self.Qx_rnd, _ = torch.qr(torch.rand(d, self.k), True)
                self.Qy_rnd, _ = torch.qr(torch.rand(d, self.k), True)
                self.Qx_rnd = self.Qx_rnd.cuda()
                self.Qy_rnd = self.Qy_rnd.cuda()
                self.Tx_rnd = torch.eye(self.k).cuda()
                self.Ty_rnd = torch.eye(self.k).cuda()
               
                self.U = alpha*self.U + (1-alpha)*self.U_rnd
                self.V = alpha*self.V + (1-alpha)*self.V_rnd
                self.Qx = alpha*self.Qx + (1-alpha)*self.Qx_rnd
                self.Qy = alpha*self.Qy + (1-alpha)*self.Qy_rnd
                self.Tx = alpha*self.Tx + (1-alpha)*self.Tx_rnd
                self.Ty = alpha*self.Ty + (1-alpha)*self.Ty_rnd
        #'''

        lr_pca = 1
        lr = 1e-1
        lr1 = 1e-1
        wwhite = 1.0
        if mode == 'train':
              x = H1
              y = H2
              
              if self.clock <= 100:
                self.Qx = self.Qx + lr_pca * x.transpose(0,1).mm(x).mm(self.Qx)
                self.Qx, _ = torch.qr(self.Qx, True)
                self.Qy = self.Qy + lr_pca * y.transpose(0,1).mm(y).mm(self.Qy)
                self.Qy, _ = torch.qr(self.Qy, True)
             
              else:
                      XXn = x.transpose(0,1).view(d, self.k, int(batch_size/self.k))
                      YYn = y.transpose(0,1).view(d, self.k, int(batch_size/self.k))
                      Vx = x.transpose(0,1).mm( - y.mm(self.Qy).mm(self.Ty)).mm(self.Tx.transpose(0,1))
                      Vy = - y.transpose(0,1).mm(x.mm(self.Qx).mm(self.Tx)).mm(self.Ty.transpose(0,1))

                      tx = torch.zeros(d,self.k).cuda()
                      ty = torch.zeros(d,self.k).cuda()
                      for j in range(XXn.shape[2]):
                        q = XXn[:,:,j]
                        tx = tx - self.invretr(self.Qx, q) / XXn.shape[2]
                        q = YYn[:,:,j]
                        ty = ty - self.invretr(self.Qy, q) / XXn.shape[2]

                      Vx = Vx + wwhite * tx
                      Vy = Vy + wwhite * ty

                      Vx = Vx - self.Qx.mm(Vx.transpose(0,1)).mm(self.Qx)
                      Vy = Vy - self.Qy.mm(Vy.transpose(0,1)).mm(self.Qy)
                      
                      Vx = Vx / torch.norm(Vx, dim=0, keepdim=True)
                      Vy = Vy / torch.norm(Vy, dim=0, keepdim=True)
                      
                      VTx = self.Qx.transpose(0,1).mm(x.transpose(0,1)).mm(- y.mm(self.Qy).mm(self.Ty))
                      VTy = - self.Qy.transpose(0,1).mm(y.transpose(0,1)).mm(x.mm(self.Qx).mm(self.Tx))
                      self.Qx = self.retrqr(self.Qx, -lr1*Vx)
                      self.Qy = self.retrqr(self.Qy, -lr1*Vy)

                      VTx = VTx - VTx.transpose(0,1)
                      VTy = VTy - VTy.transpose(0,1)
                      VTx = VTx / torch.norm(VTx, dim=0, keepdim=True)
                      VTy = VTy / torch.norm(VTy, dim=0, keepdim=True)
                      
                      VTx = 0.5 * VTx
                      VTy = 0.5 * VTy
                      self.Tx = self.Tx.mm(self.retrqr(torch.eye(self.k).cuda(), -lr*VTx))
                      self.Ty = self.Ty.mm(self.retrqr(torch.eye(self.k).cuda(), -lr*VTy))

                      Ax = self.Qx.mm(self.Tx)
                      Ay = self.Qy.mm(self.Ty)

                      nAx = Ax
                      nAy = Ay
                      
                      Du = torch.diag( 1 / torch.sqrt(torch.diag(nAx.transpose(0,1).mm(x.transpose(0,1)).mm(x).mm(nAx))))
                      Dv = torch.diag( 1 / torch.sqrt(torch.diag(nAy.transpose(0,1).mm(y.transpose(0,1)).mm(y).mm(nAy))))

                      self.U = nAx.mm(Du)
                      self.V = nAy.mm(Dv)
                    
              self.clock += 1
              print(self.clock)

        V = self.U
        U = self.V
        corr = 0
        for i in range(self.k):
            vi = V[:, i].unsqueeze(1)
            ui = U[:, i].unsqueeze(1)
            whiten1 = torch.sqrt(torch.mean(torch.matmul(H1, vi)**2))
            whiten2 = torch.sqrt(torch.mean(torch.matmul(H2, ui) ** 2))
            corr += torch.mean(torch.matmul(H1, vi) * torch.matmul(H2,ui)) / whiten1 / whiten2
        return - corr