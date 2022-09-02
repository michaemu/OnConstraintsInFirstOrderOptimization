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



def runOwnSOCP(H,c,A,b,nC,x0,lambdavec,params):
    H_sparse=csr_matrix(H);
    H_val=H_sparse.data;
    H_colidx=H_sparse.indices;
    H_rowptr=H_sparse.indptr;
    H_params=np.array([H_sparse.shape[0],H_sparse.shape[1],H_sparse.nnz]);
    
    A_sparse=csr_matrix(A);
    A_val=A_sparse.data;
    A_colidx=A_sparse.indices;
    A_rowptr=A_sparse.indptr;
    A_params=np.array([A_sparse.shape[0],A_sparse.shape[1],A_sparse.nnz]);
    
    nnC=len(nC);
    
    Atilde=np.concatenate((A[0:nC[0]+nC[1],:],np.zeros((nnC-2,len(x0)))))
    
    G=Atilde@Atilde.T;
    Gout=np.squeeze(np.reshape(G,(1,np.shape(G)[0]**2)));
    G_val=Gout.data;
    G_params=np.array([G.shape[0],G.shape[1]]);
    
    lambdavec=np.append(lambdavec,np.array([0.0,0.0])); # append 0.0 -> gives average number of constraints
    
    out=constrainedGDlib.solveSOCP(H_val,H_colidx,H_rowptr,H_params,c,\
                             A_val,A_colidx,A_rowptr,A_params,b,\
                             G_val,G_params,params,nC,x0,lambdavec);
                                   
    n_avg_constr=lambdavec[-2];
    n_avg_ineq_constr=lambdavec[-1];
    lambdavec=lambdavec[0:-2];
                       
    return out,n_avg_constr,n_avg_ineq_constr;




def generateQPdata(kappa,n,nineq,neq):
    
    lamb=np.random.uniform(1/kappa,1.0,n);
    lamb[0]=1/kappa;
    lamb[1]=1;    

    H=np.diag(lamb);
    
    c=np.random.uniform(-1.0,1.0,size=(n,1));
    A=np.random.normal(size=(nineq,n));
    b=np.zeros((nineq,1)); #np.random.normal(size=(nineq,1));
    Aeq=np.random.normal(size=(neq,n));
    beq=np.zeros((neq,1)); #np.random.normal(size=(neq,1));
    
    return H,c,A,b,Aeq,beq


def runRandomSOCP(n):
    nineq=int(np.round(n/2)); 
    neq=int(np.round(n/4));
    
    L=1.0;
    mu=0.05;
    kappa=L/mu;
    
    
    H,c,A,b,Aeq,beq=generateQPdata(kappa,n,nineq,neq)
    
    
    Asocp=np.zeros((1,n));
    Asocp=np.concatenate((Asocp,np.eye(n)));
    bsocp=np.zeros((n+1,1));
    bsocp[0]=-1;
    
    G_cvxopt=-np.concatenate((A,Asocp));
    h_cvxopt=-np.concatenate((b,bsocp));
    
    A=np.concatenate((Aeq,A,Asocp));
    b=np.concatenate((beq,b,bsocp));
    
    n_C=[neq,nineq,n+1];
    
    
    alpha=0.4/(2-.4)*(L+mu+L*(1+np.linalg.norm(np.dot(-np.linalg.inv(H),c))/np.sqrt(2)));
    
    cg=alpha+L*(1+np.linalg.norm(np.dot(-np.linalg.inv(H),c))/np.sqrt(2));
    
    T=2/(L+mu+cg) 
    MAXITER=1000;
    MAXITER_PROX=200;
    TOL_const=1e-6;
    TOL_KKT=1e-6;
    ABSTOL_PROX=1e-6;
    RELTOL_PROX=0; # not included in the paper
    omega=1.0;
    
    
    params=np.array([T,alpha,omega,TOL_const,MAXITER,TOL_KKT,MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX]);
    
    x0=np.zeros(n);
    xsol=x0.copy();
    lambdasol=np.zeros(nineq+neq+1);
    
    
    print("run own method");
    
    t1=time.perf_counter();
    out,avg_n_constr,n_avg_ineq_constr=runOwnSOCP(H,c,A,b,n_C,xsol,lambdasol,params);
    tellapsed=time.perf_counter()-t1;
    
    niterown=MAXITER;
    if out>0:
        niterown=out;
        out=1;
    
    
    
    # solve with CVXOPT
    dims={'l':n_C[1],'q':n_C[2::],'s':[]}
    
    
    t1=time.perf_counter();
    sol=solvers.coneqp(sparse(matrix(H)),matrix(c),sparse(matrix(G_cvxopt)),matrix(h_cvxopt),dims,matrix(Aeq),matrix(beq));
    tellapsedcvx=time.perf_counter()-t1;
    
    
    print("town: ",tellapsed, "success: ", out, "iter: ", niterown, "avg_n_constr: ",avg_n_constr);
    print("tcvxopt: ",tellapsedcvx, "success: ", sol['status'], "iter: ", sol['iterations']);
    
    
    outcvx=-1;
    if sol['status']=='optimal':
        outcvx=1;
    
    
    outval={'o':[tellapsed,out,niterown,avg_n_constr,n_avg_ineq_constr],\
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
        out=runRandomSOCP(nvec[k1]);
        perform2.append(out);
    perform.append(perform2);

##
savedict={'perform': perform, 'Nruns': Nruns, 'nvec': nvec, 'seed': seed};
np.save('random_tr_perf',savedict);
