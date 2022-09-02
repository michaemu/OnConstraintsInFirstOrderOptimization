import numpy as np;
import matplotlib.pyplot as plt;
from cvxopt import matrix, solvers,sparse;
from scipy.sparse import csr_matrix;

import time;


# add constrainedGDlib to path
import sys,os
filename = sys.argv[0]
abspath=os.path.abspath(filename+"/../../../constrainedGDlib/build");
sys.path.append(abspath)

import constrainedGDlib



def generateQPdata(kappa,n,nineq,neq):
    
    lamb=np.random.uniform(1/kappa,1.0,n);
    lamb[0]=1/kappa;
    lamb[1]=1;
    
    H=np.diag(lamb);
    
    c=np.random.uniform(-1.0,1.0,n);
    A=np.random.normal(size=(nineq,n));
    b=np.random.normal(size=(nineq,1));
    Aeq=np.random.normal(size=(neq,n));
    beq=np.random.normal(size=(neq,1));
    
    return H,c,A,b,Aeq,beq

def runOwnQP(H,c,A,b,Aeq,beq,x0,lambdasol,params):
    H_sparse=csr_matrix(H);
    H_val=H_sparse.data;
    H_colidx=H_sparse.indices;
    H_rowptr=H_sparse.indptr;
    H_params=np.array([H_sparse.shape[0],H_sparse.shape[1],H_sparse.nnz]);
    
    Acomb=np.concatenate([A,Aeq]);
    A_sparse=csr_matrix(Acomb);
    A_val=A_sparse.data;
    A_colidx=A_sparse.indices;
    A_rowptr=A_sparse.indptr;
    A_params=np.array([A_sparse.shape[0],A_sparse.shape[1],A_sparse.nnz]);
    
    G=Acomb@Acomb.T;
    Gout=np.squeeze(np.reshape(G,(1,np.shape(G)[0]**2)));
    G_val=Gout.data;
    G_params=np.array([G.shape[0],G.shape[1]]);
    
    neq=np.shape(Aeq)[0];
    
    bcomb=np.concatenate([b,beq]);
    
    lambdasol=np.append(lambdasol,0.0); # append 0.0 -> gives average number of constraints
    
    out=constrainedGDlib.solveQP(H_val,H_colidx,H_rowptr,H_params,c,\
                             A_val,A_colidx,A_rowptr,A_params,bcomb,\
                             G_val,G_params,params,neq,x0,lambdasol);
    n_avg_constr=lambdasol[-1];
    lambdasol=lambdasol[0:-1];

                         
    return out,n_avg_constr;


def runRandomQP(n):
    nineq=int(np.round(n/2)); 
    neq=int(np.round(n/4));
    
    
    L=1;
    mu=0.05;
    kappa=L/mu;
    
    T=2.0/(L+mu);
    alpha=0.4/T;
    
    MAXITER=1000;
    MAXITER_PROX=200;
    TOL_const=1e-6;
    TOL_KKT=1e-6;
    ABSTOL_PROX=1e-6;
    RELTOL_PROX=0;
    omega=1;
    
    params=np.array([T,alpha,omega,TOL_const,MAXITER,TOL_KKT,MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX]);
    
    H,c,A,b,Aeq,beq=generateQPdata(kappa,n,nineq,neq)
    
    xsol=np.zeros(n);
    lambdasol=np.zeros(nineq+neq);
    
    print("run own method");
    
    t1=time.perf_counter();
    out,avg_n_constr=runOwnQP(H,c,A,b,Aeq,beq,xsol,lambdasol,params);
    tellapsed=time.perf_counter()-t1;
    
    
    niterown=MAXITER;
    if out>0:
        niterown=out;
        out=1;
    
    
    t1=time.perf_counter();
    sol=solvers.qp(sparse(matrix(H)),matrix(c),matrix(-A),matrix(-b),matrix(Aeq),matrix(beq));
    tellapsedcvx=time.perf_counter()-t1;
    
    
    print("town: ",tellapsed, "success: ", out, "iter: ", niterown, "avg_n_constr: ",avg_n_constr);
    print("tcvxopt: ",tellapsedcvx, "success: ", sol['status'], "iter: ", sol['iterations']);
    

    outcvx=-1;
    if sol['status']=='optimal':
        outcvx=1;
    
    
    outval={'o':[tellapsed,out,niterown,avg_n_constr],\
            'cvx':[tellapsedcvx,outcvx,sol['iterations']]};

    return outval;


## 
seed=27111989;       
np.random.seed(seed);

    
Nruns=1;

nvec=[100,500,1000,5000,10000,20000];

perform=[];

for k1 in range(len(nvec)):
    perform2=[];
    for k2 in range(Nruns):
        print("run with n: ",nvec[k1]);
        out=runRandomQP(nvec[k1]);
        perform2.append(out);
    perform.append(perform2);

##
savedict={'perform': perform, 'Nruns': Nruns, 'nvec': nvec, 'seed': seed};
np.save('random_qp_perf',savedict);
