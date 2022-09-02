import numpy as np;
import matplotlib.pyplot as plt;
from cvxopt import matrix, solvers,sparse;
from scipy.sparse import csr_matrix;
from scipy.spatial.distance import cdist;
from scipy.sparse.linalg import eigsh;

import time;


# add constrainedGDlib to path
import sys,os
filename = sys.argv[0]
abspath=os.path.abspath(filename+"/../../../constrainedGDlib/build");
sys.path.append(abspath)

import constrainedGDlib


np.random.seed(27111989);



def generateData(ns):
    
    mean_1=0;
    mean_2=2;
    std_1=0.5;
    std_2=0.5;
    
    n1=int(ns/2);
    n2=ns-n1;
    r1=np.random.normal(mean_1,std_1,n1);
    r2=np.random.normal(mean_2,std_2,n2);
    theta1=np.random.uniform(0,2*np.pi,n1);
    theta2=np.random.uniform(0,2*np.pi,n2);
    
    x1=np.abs(r1)*np.cos(theta1);
    y1=np.abs(r1)*np.sin(theta1);
    x2=np.abs(r2)*np.cos(theta2);
    y2=np.abs(r2)*np.sin(theta2);
    
    t1=np.ones(n1);
    t2=-np.ones(n2);
    
    xt1=np.concatenate([[x1],[y1]]);
    xt2=np.concatenate([[x2],[y2]]);
    
    xt=np.concatenate((xt1,xt2),axis=1);
    t=np.concatenate((t1,t2));
    
    return xt,t


def generateQPdata(xt,t,gamma,lambda_reg,sigma):
    ns=np.shape(xt)[1];
    H=np.zeros((ns,ns));
    
    pairwise_sq_dists = cdist(xt.T,xt.T, 'sqeuclidean');
    Ker = np.exp(-pairwise_sq_dists/ (2*sigma**2));

    eval1,evec1=eigsh(Ker,1);

    
    H=Ker*np.outer(t,t)+lambda_reg*eval1*np.eye(ns);
    
    c=np.array([np.zeros(ns)]).T;
    A=np.concatenate((np.eye(ns),-np.eye(ns),np.ones((1,ns))));
    b=np.concatenate((np.zeros(ns),np.ones(ns)*(-1.0/ns)))
    b=np.array([np.append(b,gamma)]).T;
    Aeq=np.array([t]);
    beq=[[0.0]];
    
    return H,c,A,b,Aeq,beq

def generateClassifierOutput(xquery,xt,t,xsol,bsol,sigma):
    ns=np.shape(xt)[1];
    pairwise_sq_dists = cdist(xt.T,xquery[np.newaxis,:], 'sqeuclidean');
    
    Ker = np.exp(-pairwise_sq_dists/ (2*sigma**2));
    Ker = Ker*t[:,np.newaxis];
    
    y=np.dot(xsol[0:ns],Ker)+bsol;

    return y;



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


def runNuSVM(ns):
           
    sigma=1.0; # kernel width
    lambda_reg=.1; # regularization
    gamma=.1; #regularization1
    
    
    xt,t=generateData(ns);
    
    H,c,A,b,Aeq,beq=generateQPdata(xt,t,gamma,lambda_reg,sigma);
    
    
    w,v=eigsh(H,ns);
    L=w[-1];
    mu=w[0];
    
    T=2/(mu+L);
    #kappa=L/mu;
    alpha=0.4/T;
    
    MAXITER=1000;
    MAXITER_PROX=200;
    TOL_const=1e-6;
    TOL_KKT=1e-6;
    ABSTOL_PROX=1e-6;
    RELTOL_PROX=0;
    omega=1;
        
    params=np.array([T,alpha,omega,TOL_const,MAXITER,TOL_KKT,MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX]);
    
    xsol=np.zeros(ns);
    lambdasol=np.zeros(len(b)+len(beq));
    
    
    print("run own method");
    
    t1=time.perf_counter();
    out,avg_n_constr=runOwnQP(H,c,A,b,Aeq,beq,xsol,lambdasol,params);
    tellapsed=time.perf_counter()-t1;
    
    
    
    niterown=MAXITER;
    if out>0:
        niterown=out;
        out=1;
    
    
    t1=time.perf_counter();
    sol=solvers.qp(sparse(matrix(H)),matrix(c),sparse(matrix(-A)),matrix(-b),matrix(Aeq),matrix(beq));
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

nsvec=[100,500,1000,5000,10000];

perform=[];

for k1 in range(len(nsvec)):
    perform2=[];
    for k2 in range(Nruns):
        print("run with n: ",nsvec[k1]);
        out=runNuSVM(nsvec[k1]);
        perform2.append(out);
    perform.append(perform2);

##
savedict={'perform': perform, 'Nruns': Nruns, 'nsvec': nsvec, 'seed': seed};
np.save('nusvm_perf',savedict);
