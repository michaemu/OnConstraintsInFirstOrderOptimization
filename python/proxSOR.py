import numpy as np;

def proxSOR(G,d,omega,c,TOL_const,m_ineq,m_eq,MAXITER,ABSTOL,RELTOL,lambda_var):
    
    niter=MAXITER;
    if len(np.shape(m_ineq))>0:
        m_ineq=m_ineq[0];
        
    for k in range(0,MAXITER-1):
        lambdaOld=lambda_var.copy();
        
        G_lambda_cmax =0;       ## keep track of max(G*lambda+c_max), where the max is taken over all inequality constraints
        
        for j in range(len(lambda_var)):
            tmp_val=np.dot(G[j,:],lambda_var);
            lambda_var[j]=lambda_var[j]-omega/d[j]*(tmp_val+c[j]);
           
            if lambda_var[j]<=0 and j>=m_eq:
                lambda_var[j]=0;
            elif j>=m_eq and G_lambda_cmax<tmp_val+c[j]-TOL_const:  ## note that the other constraints open up -> we only care about closed constraints for max(G*lambda+cmax)
                G_lambda_cmax=tmp_val+c[j]-TOL_const;
        
        #G_lambda_cmax=np.dot(G,lambda_var) + cmax;
        #G_lambda_cmax=np.max(G_lambda_cmax[np.logical_and(lambda_var,np.array([np.arange(0,len(lambda_var))]).T>=m_eq).flatten()]);
        if np.sqrt(sum((lambda_var-lambdaOld)**2)) <= ABSTOL+RELTOL*np.sqrt(sum(lambdaOld**2)) and G_lambda_cmax <= 0:
            niter=k+1;
            break;
    
    return niter,lambda_var;



def proxSORvarStep(G,d,omega,c,cmax,m_ineq,m_eq,MAXITER,ABSTOL,RELTOL,lambda_var):
    
    niter=MAXITER;
    if len(np.shape(m_ineq))>0:
        m_ineq=m_ineq[0];
        
    for k in range(0,MAXITER-1):
        lambdaOld=lambda_var.copy();
        
        G_lambda_cmax =0;       ## keep track of max(G*lambda+c_max), where the max is taken over all inequality constraints
        
        for j in range(len(lambda_var)):
            tmp_val=np.dot(G[j,:],lambda_var);
            lambda_var[j]=lambda_var[j]-omega/d[j]*(tmp_val+c[j]);
           
            if lambda_var[j]<=0 and j>=m_eq:
                lambda_var[j]=0;
            elif j>=m_eq and G_lambda_cmax<tmp_val+cmax[j]:  ## note that the other constraints open up -> we only care about closed constraints for max(G*lambda+cmax)
                G_lambda_cmax=tmp_val+cmax[j];
        
        v=lambda_var-lambdaOld;
        
        tmp=-lambdaOld/v;
        tmp=tmp[m_eq:-1]; # pick only inequality constraints
        tmp=tmp[tmp>0]; # ensure that these can be violated at all
        tmp2=-1.0/np.dot(v.T,np.dot(G,v))*(np.dot(np.dot(lambdaOld.T,G),v)+np.dot(c.T,v));
        tmp2=tmp2[0,0];
        
        tau_star=1.0;
        if len(tmp)>0:
            tau_star=np.min([np.min(tmp),tmp2]);
        elif tmp2>0:
            tau_star=tmp2;
        
        #print('iter: '+str(k)+' step size: '+str(tau_star));
        
        lambda_var=lambdaOld+tau_star*v;
        
        if np.sqrt(sum((lambda_var-lambdaOld)**2)) <= ABSTOL+RELTOL*np.sqrt(sum(lambdaOld**2)): # and G_lambda_cmax <= 0:
            niter=k+1;
            break;
    
    return niter,lambda_var;
