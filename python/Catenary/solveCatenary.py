import numpy as np;
#import functools;

import sys
sys.path.append('../')


from proxSOR import proxSOR;

# compute equality constraints
def getGeq(z,z0,z1):
    N=len(z)/2+1;
    zaug=np.array([np.concatenate((z0.flatten(),z.flatten(),z1.flatten()))]).T;
    g=(zaug[0:-2:2]-zaug[2::2])**2+(zaug[1:-1:2]-zaug[3::2])**2-4/(N**2);
    return g;

# compute gradient of inequality constraints
def getGradG(z,z0,z1):
    N=int(np.round(len(z)/2+1));
    
    dg=np.zeros((N,len(z)));
    
    for k in range(N):
        
        if k==0:
            idx=0;
            idy=1;
            dg[0,0]=-2.0*(z0[0]-z[idx]);
            dg[0,1]=-2.0*(z0[1]-z[idy]);
            
        elif k==N-1:
            idx=2*(N-1)-2;
            idy=2*(N-1)-1;
            dg[N-1,idx]=2.0*(z[idx]-z1[0]);
            dg[N-1,idy]=2.0*(z[idy]-z1[1]);
            
        else:
            idx=2*(k-1);
            idy=2*(k-1)+1;
            idx2=2*k;
            idy2=2*k+1;
            
            dg[k,idx]=2.0*(z[idx]-z[idx2]);
            dg[k,idy]=2.0*(z[idy]-z[idy2]);
            dg[k,idx2]=-2.0*(z[idx]-z[idx2]);
            dg[k,idy2]=-2.0*(z[idy]-z[idy2]);
            
    return dg;

# compute inequality constraints
def getGineq(z,z2,r0):
    g=(z[0::2]-z2[0])**2+(z[1::2]-z2[1])**2-r0**2;
    return g;

# gradient of inequality constraints
def getGradGineq(z,z2,r0):
    N=int(np.round(len(z)/2+1))-1;
    
    dg=np.zeros((N,len(z)));
    
    for k in range(N):
        idx=2*k;
        idy=2*k+1;
        dg[k,idx]=2*(z[idx]-z2[0]);
        dg[k,idy]=2*(z[idy]-z2[1]);
    
    return dg;
    
# function used to numerically calculate gradients -> used for debugging
def numdf(f_,x,dx):
    n=np.shape(x)[0];
    y=np.eye(n)*dx;
    m=np.shape(f_(x))[0];
    
    df=np.zeros((m,n));
    for k in range(n):
        tmp=(f_(x+np.array([y[:,k]]).T)-f_(x))/dx;
        df[:,k]=tmp.flatten();
    
    return df;
    
  
# solves the catenary problem
# --------------------------------------------------------    
# note that x contains the (x,y)-coordinates of the joints
# x=(x_1,y_1,x_2,y_2,x_3,y_3, ... )
# N is the number of links in the chain
# z0 left boundary condition
# z1 right boundary condition
# z2 r0 defines inequality constraint
# --------------------------------------------------------
# outputs whole solution trajectory, cost function trajectory, trajectory of multipliers and 
# trajectory of constraints
def solveCatenary(N,x0,T,alpha,TOL_const,omega,\
                  TOL_KKT,MAXITER,\
                  MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX,z0,z1,z2,r0):
    n=len(x0);
    
    # number of equality/inequality constraints
    neq=N;
    nineq=N-1;
    
    
    # define the objective function (potential energy)
    c=np.zeros((n,1));
    c[1::2]=9.81/(N+1)*np.ones((N-1,1));
    
    
    idx_eq=np.array([[True]*neq+[False]*nineq]).T;
    
    xsol=np.zeros((MAXITER,n));
    fsol=np.zeros((MAXITER,1));
    lambdasol=np.zeros((MAXITER,nineq+neq));
    prox_iter=np.zeros((MAXITER,1));
    g_ineq_tot=np.zeros((MAXITER,nineq));
    g_eq_tot=np.zeros((MAXITER,neq));
    n_act_constr=np.zeros((MAXITER,1));
    
    xsol[0,:]=x0.T;
    lambdasol[0,:]=np.zeros((1,neq+nineq));
    fsol[0]=np.dot(c.T,x0);
    
    for k in range(1,MAXITER):
        x1=np.array([xsol[k-1,:].copy()]).T;
        
        # evaluate gradient
        df=c;

        # evaluate constraints equality and inequality
        g_eq=getGeq(x1,z0,z1);
        g_ineq=getGineq(x1,z2,r0);        
        
        g=np.concatenate((g_eq,g_ineq));
        
        
        # check which constraints are active
        # -> equality constraints are always included
        idx=np.logical_or(g<=TOL_const,idx_eq);
        
        g_ineq_tot[k-1,:]=g_ineq.T;
        g_eq_tot[k-1,:]=g_eq.T;
        
        # check whether there are active constraints
        if sum(idx)>0:
            nineq_tmp=sum(g_ineq<=TOL_const);

            # evaluate gradients            
            W1=getGradG(x1,z0,z1).T;
            W2=getGradGineq(x1,z2,r0).T;
            W=np.concatenate((W1,W2),axis=1);
            
            # include only the constraints which are active
            Wtmp=W[:,idx.flatten()];
            Gtmp=np.dot(Wtmp.T,Wtmp);
                            
            # create a good initial guess for lambda
            # this is only included in Python, since the implementation of the prox 
            # function is slow here
            lambdatmp=np.dot(np.linalg.pinv(Gtmp),np.dot(df.T,Wtmp).T-alpha*g[idx.flatten()]);

            # compute the multipliers lambda
            itertmp,lambdatmp=proxSOR(Gtmp,np.diag(Gtmp),omega,-np.dot(df.T,Wtmp).T+alpha*g[idx.flatten()],TOL_const*alpha/2.0,nineq_tmp,neq,\
                                    MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX,lambdatmp);
            
            prox_iter[k-1]=itertmp;
            lambdasol[k,idx.flatten()]=lambdatmp.T;
            n_act_constr[k-1]=nineq_tmp;    # keep track of how many constraints entered at this iteration
                        
            # perform update
            x2=x1-T*df+T*np.dot(Wtmp,lambdatmp);
            
        else:
            lambdasol[k,:]=np.zeros(shape=(1,neq+nineq));
            prox_iter[k-1]=0;
            n_act_constr[k-1]=0;

            # perform update -> here no constraints are active
            x2=x1-T*df;
            
        xsol[k,:]=x2.T;
        fsol[k]=np.dot(c.T,x2);
                
        if np.max(np.abs(x1-x2)) <= T*TOL_KKT:
            break;

        
    niter=k;
    return xsol[0:k+1,:],fsol[0:k+1],lambdasol[0:k+1,:],niter,prox_iter[0:k],g_ineq_tot[0:k,:],g_eq_tot[0:k,:],n_act_constr[0:k];
