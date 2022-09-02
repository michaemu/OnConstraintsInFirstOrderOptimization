import numpy as np;
from proxSOR import proxSOR;
from proxSOR import proxSORvarStep;


def QP(H,c,A,b,Aeq,beq,
                  x0,T,alpha,TOL_const,omega,
                  TOL_KKT,MAXITER,MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX):

    n=np.shape(H)[0];
    m_eq=np.shape(Aeq)[0];
    m_ineq=np.shape(A)[0];
    
    idx_eq=np.array([[True]*m_eq+[False]*m_ineq]).T;
    
    W=np.transpose(np.concatenate((Aeq,A)));
    G=np.dot(np.transpose(W),W);
    
    xsol=np.zeros(shape=(MAXITER,n));
    lambdasol=np.zeros(shape=(MAXITER,m_eq+m_ineq));
    prox_iter=np.zeros(shape=(MAXITER,1));
    g_ineq_tot=np.zeros(shape=(MAXITER,m_ineq));
    g_eq_tot=np.zeros(shape=(MAXITER,m_eq));
    fsol=np.zeros(shape=(MAXITER,1));
    n_act_constr=np.zeros(shape=(MAXITER,1));
    
    " add initial conditions "
    xsol[0,:]=x0.T;
    lambdasol[0,:]=np.zeros(shape=(1,m_eq+m_ineq));
    fsol[0]=1/2*np.dot(np.dot(H,x0).T,x0)+np.dot(c.T,x0);
    
    for k in range(1,MAXITER):
        x1=np.array([xsol[k-1,:].copy()]).T;
        df=np.dot(H,x1)+c;
        
        g_ineq=np.dot(A,x1)-b;
        g_eq=np.dot(Aeq,x1)-beq;
        g=np.concatenate((g_eq,g_ineq));
        
        idx=np.logical_or(g<=TOL_const,idx_eq);    "always activate equality constraints"
        
        g_ineq_tot[k-1,:]=g_ineq.T;
        g_eq_tot[k-1,:]=g_eq.T;
        
        if sum(idx)>0:
            m_ineq_tmp=sum(g_ineq<=TOL_const);
            
            Wtmp=W[:,idx.flatten()];
            Gtmp=G[:,idx.flatten()][idx.flatten(),:];
            
            lambdatmp=np.array([lambdasol[k-1,:].copy()]).T;

            itertmp,lambdatmp[idx.flatten()]=proxSOR(Gtmp,np.diag(Gtmp),omega,-np.dot(df.T,Wtmp).T+alpha*g[idx.flatten()],TOL_const*alpha/2.0,m_ineq_tmp,m_eq,\
                                    MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX,lambdatmp[idx.flatten()]);
            
            prox_iter[k-1]=itertmp;
            lambdasol[k,:]=lambdatmp.T;
            n_act_constr[k-1]=m_ineq_tmp;
                        
            x2=x1-T*df+T*np.dot(Wtmp,lambdatmp[idx.flatten()]);
            
        else:
            lambdasol[k,:]=np.zeros(shape=(1,m_eq+m_ineq));
            prox_iter[k-1]=0;
            n_act_constr[k-1]=0;

            x2=x1-T*df;
            
        xsol[k,:]=x2.T;
        fsol[k]=1/2*np.dot(np.dot(H,x2).T,x2)+np.dot(c.T,x2);
                
        if np.max(np.abs(x1-x2)) <= T*TOL_KKT:
            break;

    
    niter=k;
    return xsol[0:k+1,:],fsol[0:k+1],lambdasol[0:k+1,:],niter,prox_iter[0:k],g_ineq_tot[0:k,:],g_eq_tot[0:k,:],n_act_constr[0:k];






def SOCP(H,c,A,b,n_C,
                  x0,T,alpha,TOL_const,omega,
                  TOL_KKT,MAXITER,MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX):

    n=np.shape(H)[0];
    m_eq=n_C[0];
    if len(n_C)>1:
        m_ineq=n_C[1]+len(n_C)-2;
    else:
        m_ineq=0;
    
    
    idx_eq=np.array([[True]*m_eq+[False]*m_ineq]).T;
    
    xsol=np.zeros(shape=(MAXITER,n));
    lambdasol=np.zeros(shape=(MAXITER,m_eq+m_ineq));
    prox_iter=np.zeros(shape=(MAXITER,1));
    g_ineq_tot=np.zeros(shape=(MAXITER,m_ineq));
    g_eq_tot=np.zeros(shape=(MAXITER,m_eq));
    fsol=np.zeros(shape=(MAXITER,1));
    n_act_constr=np.zeros(shape=(MAXITER,1));
    Gcond=np.zeros(shape=(MAXITER,1));

    
    " add initial conditions "
    xsol[0,:]=x0.T;
    lambdasol[0,:]=np.zeros(shape=(1,m_eq+m_ineq));
    fsol[0]=1/2*np.dot(np.dot(H,x0).T,x0)+np.dot(c.T,x0);
    
    for k in range(1,MAXITER):
        x1=np.array([xsol[k-1,:].copy()]).T;
        df=np.dot(H,x1)+c;
        
        g_tmp=np.dot(A,x1)-b;
        g_eq=g_tmp[0:n_C[0]];
        g_ineq=np.zeros((m_ineq,1));
        g_ineq[0:n_C[1]]=g_tmp[n_C[0]:n_C[0]+n_C[1]];
        idxtmp=n_C[0]+n_C[1];
        for j in range(2,len(n_C)):
            g_ineq[n_C[1]+j-2]=-np.linalg.norm(g_tmp[idxtmp+1:idxtmp+n_C[j]]) + g_tmp[idxtmp];
            idxtmp=idxtmp+n_C[j];
        
        g=np.concatenate((g_eq,g_ineq));
        
        idx=np.logical_or(g<=TOL_const,idx_eq);    "always activate equality constraints"
        
        g_ineq_tot[k-1,:]=g_ineq.T;
        g_eq_tot[k-1,:]=g_eq.T;
        
        if np.sum(idx)>0:
            
            idx_1=np.concatenate((idx[0:n_C[0]+n_C[1]],[[False]]*(np.shape(A)[0]-n_C[0]-n_C[1])));
            Wtmp=A[idx_1.flatten(),:];
            
            idx_2=n_C[0]+n_C[1];
            for j in range(2,len(n_C)):
                if g[n_C[0]+n_C[1]+j-2]<=TOL_const:
                    v=g_tmp[idx_2+1:idx_2+n_C[j]];
                    if np.linalg.norm(v)==0:
                        #print('problem with norm(v)=0');
                        vtmp=np.concatenate(([[1]],-v)).T;
                        wtmp=np.dot(vtmp,A[idx_2:idx_2+n_C[j],:]);
                    else:
                        vtmp=np.concatenate(([[1]],-v/np.linalg.norm(v))).T;
                        wtmp=np.dot(vtmp,A[idx_2:idx_2+n_C[j],:]);
                    Wtmp=np.concatenate((Wtmp,wtmp));
                idx_2=idx_2+n_C[j];
            
            Wtmp=Wtmp.T;
            Gtmp=np.dot(Wtmp.T,Wtmp);
            
            lambdatmp=np.array([lambdasol[k-1,:].copy()]).T;
            itertmp,lambdatmp[idx.flatten()]=proxSOR(Gtmp,np.diag(Gtmp),omega,-np.dot(df.T,Wtmp).T+alpha*g[idx.flatten()],TOL_const*alpha/2.0,np.sum(idx)-m_eq,m_eq,\
                                    MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX,lambdatmp[idx.flatten()]);
            
            prox_iter[k-1]=itertmp;
            lambdasol[k,:]=lambdatmp.T;
            n_act_constr[k-1]=np.sum(idx)-m_eq;
            Gcond[k-1]=np.linalg.cond(Gtmp);
                        
            x2=x1-T*df+T*np.dot(Wtmp,lambdatmp[idx.flatten()]);
            
        else:
            lambdasol[k,:]=np.zeros(shape=(1,m_eq+m_ineq));
            prox_iter[k-1]=0;
            n_act_constr[k-1]=0;
            Gcond[k-1]=1;

            x2=x1-T*df;
            
        xsol[k,:]=x2.T;
        fsol[k]=1/2*np.dot(np.dot(H,x2).T,x2)+np.dot(c.T,x2);
                
        if np.max(np.abs(x1-x2)) <= T*TOL_KKT:
            break;

    
    niter=k;
    return xsol[0:k+1,:],fsol[0:k+1],lambdasol[0:k+1,:],niter,prox_iter[0:k],g_ineq_tot[0:k,:],g_eq_tot[0:k,:],n_act_constr[0:k],Gcond[0:k];
