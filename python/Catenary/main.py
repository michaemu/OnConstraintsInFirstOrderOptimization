
import numpy as np;
import solveCatenary;
import matplotlib.pyplot as plt;



# plotting function
def drawCatenary(fighandle,idx,xsol,z0,z1,z2,r0,addcircle,color):
    xsolaug=np.array([np.concatenate((z0.flatten(),xsol[idx,:],z1.flatten()))]).T;
    
    xout=xsolaug[0::2];
    yout=xsolaug[1::2];
    
    plt.figure(fighandle)
    if addcircle==True:
        varphi=np.linspace(0,np.pi*2.0,500);
        xcircle=z2[0]+np.cos(varphi)*r0;
        ycircle=z2[1]+np.sin(varphi)*r0;
        plt.plot(xcircle,ycircle,'k')
    
    colorstring=color+'.-';
    plt.plot(xout,yout,colorstring)
    plt.axis('equal')
    


plt.close('all')

N=20; # N is number of links, N+1 number of joints including z0,z1;
      # N needs to be even


z0=np.array([[0,0]]).T;         # left fixation
z1=np.array([[1,0]]).T;         # right fixation
z2=np.array([[0.5,-.8]]).T;     # center of ball
r0=0.5;                         # radius of ball

np.random.seed(27111989);       # set seed
x0=np.random.normal(size=((N-1)*2,1))*.05;          # random initial y values
x0[0:-1:2]=np.array([np.arange(1./N,1.,1./N)]).T;   # equidistant x values

# parameters for solver
TOL_KKT=1e-6;
TOL_const=1e-6;
MAXITER=25000;
MAXITER_PROX=1000;
ABSTOL_PROX=1e-12;
RELTOL_PROX=0;
omega=1;
T=2.0/(9.81*N); # cg~9.81*N
alpha=0.8/T;


# compute local solutions to the NLP
xsol,fsol,lambdasol,niter,prox_iter,g_ineq_tot,g_eq_tot,n_act_constr=solveCatenary.solveCatenary(N,x0,T,alpha,\
                                                                                                 TOL_const,omega,TOL_KKT,\
                                                                                                 MAXITER,MAXITER_PROX,ABSTOL_PROX,RELTOL_PROX,z0,z1,z2,r0);

print("optimization terminated in "+str(niter)+" iterations.")


err_norm_g_ineq =np.zeros(shape=(np.shape(g_ineq_tot)[0],1))
for j in range(np.shape(g_ineq_tot)[0]):
    err_norm_g_ineq[j]=np.sqrt(np.sum(g_ineq_tot[j,g_ineq_tot[j,:]<=0]**2));

err_norm_g_eq = np.sqrt(np.sum(g_eq_tot**2,axis=1));




drawCatenary(1,0,xsol,z0,z1,z2,r0,False,'k')
drawCatenary(1,100,xsol,z0,z1,z2,r0,False,'b')
drawCatenary(1,1000,xsol,z0,z1,z2,r0,False,'g')
drawCatenary(1,-1,xsol,z0,z1,z2,r0,True,'r')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(['initial','k=100','k=1000','constraint','final'])



plt.figure()
plt.subplot(211)
plt.loglog(err_norm_g_eq)
plt.xlabel('iterations')
plt.ylabel('$|h(x_k)|$')

plt.subplot(212)
plt.plot(fsol,'k')
plt.xlabel('iterations')
plt.ylabel('cost')

#tikzplotlib.save("catenary_2.tikz", axis_width='\\figurewidth', axis_height ='\\figureheight')
