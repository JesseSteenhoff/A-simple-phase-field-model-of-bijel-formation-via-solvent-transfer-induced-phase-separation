# -*- coding: utf-8 -*-
"""
Analysis of Bijel Formation Dynamics During Solvent Transfer-Induced Phase Separation Using Phase-Field Simulations
Model Validation  
@author: J.M. Steenhoff
"""
#Import all the required modules 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scienceplots
from time import process_time
from matplotlib.colors import ListedColormap
from scipy.optimize import fsolve

#Format style and dpi of all Figures 
plt.rcParams['figure.dpi'] = 300
plt.style.use(['science','no-latex'])

#%% Create a custom magenta-black colourmap

#RBG value maximum (exclusive) 
N_map=256

#RBG values for the colours in the colourmap 
Colour_top=[233,98,233]              
Colour_centre=[98,12,82]
Colour_bottom=[0,0,0]

#Centre position of the colourmap
N_mapC=int(N_map/2)

#RGBA colourmap template. Last column is for the alpha-value (here set to 1)
MagentaBlack=np.ones((N_map,4))        

#Creates the custom colourmap 
for i in range(3):
    MagentaBlack[0:N_mapC,i]=np.linspace(Colour_bottom[i]/N_map,Colour_centre[i]/N_map,N_mapC)   
    MagentaBlack[N_mapC:N_map,i]=np.linspace(Colour_centre[i]/N_map,Colour_top[i]/N_map,N_mapC)

MagentaBlack=ListedColormap(MagentaBlack)   

#%% Define the different classes for the model. Effectively blueprints for phase-field (PF) objects

#The 'Liquid' class creates a PF object that evolves in accordance with the Cahn-Hilliard equation 
class Liquid:
    
    #Binary interaction parameter
    chi=2.5                                   
    
    #Field gradient penalty coefficients 
    kappa=1.0
    
    #Relative mobility with respect to solvent 
    M=1
    
    #Simulation timestep 
    dt=0.01
    #Stencil spacing 
    h=1
    
    #Initialisation method that creates a field  (Size[0]xSize[1]) of composition phi0, including some random thermal noise
    def __init__(self,Size,phi0):
        self.Size=Size
        self.phi0=phi0
        self.phi=self.phi0+np.random.randint(-10,10,(self.Size[0],self.Size[1]))*0.001
        
        #Solve the binodal equation to find the equilibrium compositions 
        self.phimin=fsolve(lambda x: np.log(x/(1-x))+self.chi*(1-2*x),0.01)
        self.phimax=1-self.phimin
    
    #Method that plots the current state of the field, normalised with respect to the equilibrium compositions 
    def Show(self,C):                           
        plt.figure()
        
        Fontsize=10
        plt.xlabel('Horizontal position, $\hat{x}$',fontsize=Fontsize)
        plt.ylabel('Vertical position, $\hat{y}$',fontsize=Fontsize)
        plt.tick_params(axis='both',labelsize=Fontsize)
        
        plt.imshow(self.phi,norm=clr.Normalize(vmin=self.phimin,vmax=self.phimax),cmap=C)
        
        Colorbar=plt.colorbar()
        Colorbar.ax.tick_params(labelsize=Fontsize)
        Colorbar.set_label('$\phi$',fontsize=Fontsize)
        
        return None
    
    #Method that calculates the Laplacian of an input field 'F' via central finite-difference (5-point stencil)
    def Calc_Laplacian(self,F):               
        
        #Dimensions of the input field 
        Ny,Nx=F.shape
        
        #Create templates for the Laplacian contributions  
        Laplacian_x=np.zeros(F.shape)
        Laplacian_y=np.zeros(F.shape)
        
        #Apply periodic boundary conditions along the x-direction 
        Laplacian_x[:,1:Nx-1]=(-2*F[:,1:Nx-1]+F[:,2:Nx]+F[:,0:Nx-2])/(self.h**2)
        Laplacian_x[:,0]=(-2*F[:,0]+F[:,1]+F[:,Nx-1])/(self.h**2)                              
        Laplacian_x[:,Nx-1]=(-2*F[:,Nx-1]+F[:,Nx-2]+F[:,0])/(self.h**2)
        
        #Apply periodic boundary conditions along the y-direction
        Laplacian_y[1:Ny-1,:]=(-2*F[1:Ny-1,:]+F[2:Ny,:]+F[0:Ny-2,:])/(self.h**2)
        Laplacian_y[0,:]=(-2*F[0,:]+F[1,:]+F[Ny-1,:])/(self.h**2)                            
        Laplacian_y[Ny-1,:]=(-2*F[Ny-1,:]+F[Ny-2,:]+F[0,:])/(self.h**2)                
        
        Laplacian=Laplacian_x+Laplacian_y
        
        return Laplacian
    
    #Method that calculates the gradient vector for an input field 'F' via central finite-difference (5-point stencil)
    def Calc_Gradient(self,F):            
        
        #Dimensions of the input field 
        Ny,Nx=F.shape
        
        #Create templates for the gradient vector components  
        Gradientx=np.zeros(F.shape)                                                     
        Gradienty=np.zeros(F.shape)
        
        #Apply periodic boundary conditions along the x-direction 
        Gradientx[:,1:Nx-1]=(F[:,2:Nx]-F[:,0:Nx-2])/(2*self.h)
        Gradientx[:,0]=(F[:,1]-F[:,Nx-1])/(2*self.h)                                                        
        Gradientx[:,Nx-1]=(F[:,0]-F[:,Nx-2])/(2*self.h)
        
        #Apply periodic boundary conditions along the y-direction 
        Gradienty[1:Ny-1,:]=(F[2:Ny,:]-F[0:Ny-2,:])/(2*self.h)
        Gradienty[0,:]=(F[1,:]-F[Ny-1,:])/(2*self.h)                                  
        Gradienty[Ny-1,:]=(F[0,:]-F[Ny-2,:])/(2*self.h)        
        
        #Gradient vector 
        Gradient=(Gradientx,Gradienty)
        
        return Gradient
    
    #Method that evolves the order parameter field according to the nondimensionalised Cahn-Hilliard equation 
    def Propagate(self,phi):
        
        #Calculates the functional derivative of the free energy (chemical potential) wrt. oil component (phi)
        Mu=np.log(phi/(1-phi))+self.chi*(1-2*phi)-self.kappa*self.Calc_Laplacian(phi)
        
        #Update the order parameter field 
        self.phi+=(self.M*self.Calc_Laplacian(Mu))*self.dt
    
        return self.phi
    
    #Method that returns the current system free energy from the free energy functional 
    def Calc_FreeEnergy(self,phi):
         
        #Find the gradient vector 
        Gradient_phi=self.Calc_Gradient(phi)
        
        #Calculate the bulk and interfacial energy contributions 
        FreeEnergy_Bulk=np.sum(phi*np.log(phi)+(1-phi)*np.log(1-phi)+self.chi*phi*(1-phi))
        FreeEnergy_Interface=np.sum(self.kappa/2*(Gradient_phi[0]**2+Gradient_phi[1]**2))
        
        FreeEnergy=FreeEnergy_Bulk+FreeEnergy_Interface
        
        return (FreeEnergy_Bulk,FreeEnergy_Interface,FreeEnergy)
    
#Function used for seeding the system with composition nucleii (Field (I), Position (r), Radius (R) and Composition (C))
def Nucleate(I,r,R,C):
    a=0.5
    for j in np.arange(-R,R+1,1): 
        y=j
        for i in np.arange(-R,R+1,1):
            x=i
            if np.sqrt(x**2+y**2)<R-a:
                I[r[0]+i,r[1]+j]=C
    return I

    
#%% Run a PF simulation of STrIPS 

#Simulation dimensions 
Dimensions=(200,200)

#Initial composition (oil-component)
phi0=0.50

#Initialise the 'oil' order parameter field
Oil=Liquid(Dimensions,phi0)

#Seed the initial liquid field with nucleation sites
 
#Composition of the nucleii
if phi0<0.50:
    phi_n=Oil.phimax
else:
    phi_n=Oil.phimin

#Number of nucleii
Nc=0

#Minimal and maximal nucleii radii
Rmin,Rmax=(3,5)

#Generate selection of uniformly distributed nucleii radii
Rc=np.random.randint(Rmin,Rmax,size=Nc)

#Generate selection of uniformly distributed nucleii positions 
xc=np.random.randint(Rmax,Dimensions[0]-Rmax,size=Nc)
yc=np.random.randint(Rmax,Dimensions[0]-Rmax,size=Nc)

#Place the nucleii
for i in range(len(xc)):
    Nucleate(Oil.phi,(xc[i],yc[i]),Rc[i],phi_n)

#Storage list of the system free energy
SystemEnergy=[]

#Add the initial energy of the system 
SystemEnergy.append(Oil.Calc_FreeEnergy(Oil.phi))

#Total number of simulation steps 
N=25000                     

#Start-point for determining computation time 
t0=process_time()  

for n in range(N+1):
    
    #Propagate the fields in time via the Cahn-Hilliard equation / Fick's second law
    Oil.Propagate(Oil.phi)

    #Calculate and return the new free energy of the system
    SystemEnergy.append(Oil.Calc_FreeEnergy(Oil.phi))
    
    #Plot the oil-field at 10% intervals 
    if n%(N/10)==0:                
        Oil.Show(C=MagentaBlack)
        print(str(100*n/N)+'% complete')

#End-point for determining computation time
t1=process_time()   
#Total computation time
Deltat=t1-t0         
print('Computation time: '+str(Deltat)+' s')

#%% Visualise the evolution of the bulk, interfacial and total free energy of the system 

Bulk=[]
Interface=[]
Total=[]

for i in range(len(SystemEnergy)):
    Bulk.append(SystemEnergy[i][0])
    Interface.append(SystemEnergy[i][1])
    Total.append(SystemEnergy[i][2])

plt.figure()
plt.minorticks_on()
plt.xlabel('Simulated time, $\hat{t}$')
plt.ylabel('System energy, $\hat{E}$')

#Simulated time (reduced units)
t_sim=np.arange(0,N+2,1)*Oil.dt

plt.plot(t_sim,Total,label='Total free energy')
plt.plot(t_sim,Bulk,label='Bulk energy')
plt.plot(t_sim,Interface,label='Interfacial energy')

plt.legend(frameon='')

#%% Export the final morphology to a .txt image

#Path=''
#np.savetxt(Path, Oil.phi)
