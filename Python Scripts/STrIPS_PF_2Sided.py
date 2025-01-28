# -*- coding: utf-8 -*-
"""
Analysis of Bijel Formation Dynamics During Solvent Transfer-Induced Phase Separation Using Phase-Field Simulations
STrIPS-simulation 2-sided 
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
from scipy.integrate import simpson 

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
    chi=3.0                                   
    
    #Field gradient penalty coefficients 
    kappa=0.5
    
    #Relative mobility with respect to solvent 
    M=0.01
    
    #Simulation timestep 
    dt=0.50
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
        plt.imshow(self.phi,norm=clr.Normalize(vmin=self.phimin,vmax=self.phimax),cmap=C)
        plt.colorbar()
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
        
        #Apply zero-flux boundary conditions at the top and bottom in the y-direction 
        Laplacian_y[1:Ny-1,:]=(-2*F[1:Ny-1,:]+F[2:Ny,:]+F[0:Ny-2,:])/(self.h**2)
        Laplacian_y[0,:]=(-1*F[0,:]+F[1,:])/(self.h**2)                            
        Laplacian_y[Ny-1,:]=(-1*F[Ny-1,:]+F[Ny-2,:])/(self.h**2)              
        
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
        
        #Apply no-flux conditions along the y-direction 
        Gradienty[1:Ny-1,:]=(F[2:Ny,:]-F[0:Ny-2,:])/(2*self.h)
        Gradienty[0,:]=(F[1,:]-F[0,:])/(2*self.h)                                  
        Gradienty[Ny-1,:]=(F[Ny-1,:]-F[Ny-2,:])/(2*self.h)        
        
        #Gradient vector 
        Gradient=(Gradientx,Gradienty)
        
        return Gradient
    
    #Method that calculates the solvent-dependent interaction parameter based on a linear relationship. 
    def Calc_Chi(self,phis):
        
        #Interaction parameter and solvent order parameter at the critical point (first one is known, second one is set)
        chi_c=2.0
        phis_c=0.50
        
        return self.chi-(self.chi-chi_c)*phis/phis_c
    
    #Method that evolves the order parameter field according to the nondimensionalised Cahn-Hilliard equation 
    def Propagate(self,phi,chi):
        
        #Calculates the functional derivative of the free energy (chemical potential) wrt. oil component (phi)
        Mu=np.log(phi/(1-phi))+chi*(1-2*phi)-self.kappa*self.Calc_Laplacian(phi)
        
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
    
    #Method that calculates the local interfacial tension (IFT) by taking the geometric mean of the horizontal and vertical contributions 
    def Calc_IFT(self,phi):
        
        #Calculate the gradients in phi in perpendicular directions 
        Gradx,Grady=self.Calc_Gradient(phi)
        
        #Calculate the perpendicular contributions of the interfacial tension through integration of the squared gradient 
        sigmax=self.kappa*simpson(Gradx**2,axis=1)
        sigmay=self.kappa*simpson(Grady**2,axis=0)
        
        #Create a field where each element is the geometric mean of the interfacial tensions corresponding to its row (x) and column (y) 
        sigma=np.sqrt((np.ones(phi.shape)*np.atleast_2d(sigmax).transpose())*sigmay)
        
        return sigma
    
    #Method that arrests the phase-separation after a critical value of the interfacial tension has been reached (by lowering the mobility) 
    def Arrest(self,phi):
        
        #Critical value of the interfacial tension 
        sigma_c=1
        
        Check=np.copy(self.Calc_IFT(phi))
        
        #Critical values higher and lower than 1 are dealt with appropriately 
        if sigma_c>=1:
            Check[Check<sigma_c]=1
            Check[Check>sigma_c]=10**-6
        
        if sigma_c<1:
            Check2=np.copy(Check)
            Check[Check<sigma_c]=1
            Check[Check2>sigma_c]=10**-6
        
        self.M=self.M*Check
    
        return self.M


#The 'solvent' class creates a PF object that evolves in accordance with Fick's second law  
class solvent:
    
    #Stencil spacing 
    h=1
    
    #Solvent order parameter boundary condition (Top)
    BC_T=0.00
    #Solvent order paramater boundary condition (Bottom)
    BC_B=0.00
    
    #Initialisation method that creates a field  (Size[0]xSize[1]) of composition phi0. 
    def __init__(self,Size,phi0,dt):
        self.Size=Size
        self.phi0=phi0
        self.dt=dt
        self.phi=np.zeros((Size[0],Size[1]))+self.phi0      
    
    #Method that plots the current state of the field, normalised with respect to its extremum values
    def Show(self,C):                           
        plt.figure()
        plt.imshow(self.phi,norm=clr.Normalize(vmin=0,vmax=self.phi0),cmap='Blues')
        plt.colorbar()
        return None
    
    #Method that plots a profile of the order parameter through a central cross-section of the field 
    def Profile(self):
        n_half=int(self.Size[0]/2)
        plt.figure()
        plt.xlabel('Vertical position, $\hat{y}$')
        plt.ylabel('Solvent order parameter, $\phi_s$')
        plt.plot(self.phi[:,n_half])
    
    #Method that calculates the Laplacian of the solvent field, which uses different (flux) boundary conditions (5-point stencil). The flux-boundary condition is set up with the BC Dirichlet condition. 
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
        
        #Apply flux boundary condition at top and bottom in the y-direction 
        Laplacian_y[1:Ny-1,:]=(-2*F[1:Ny-1,:]+F[2:Ny,:]+F[0:Ny-2,:])/(self.h**2)
        Laplacian_y[0,:]=(-2*F[0,:]+F[1,:]+self.BC_T)/(self.h**2)                            
        Laplacian_y[Ny-1,:]=(-2*F[Ny-1,:]+F[Ny-2,:]+self.BC_B)/(self.h**2)              
        
        Laplacian=Laplacian_x+Laplacian_y
        
        return Laplacian
    
    #Method that propagates the solvent order parameter field in time according to Fick's second law (nondimensionalised)
    def Propagate(self,phi):
        #Update the order parameter field 
        self.phi+=(self.Calc_Laplacian(phi))*self.dt
    
        return self.phi
    

#%% Run a PF simulation of STrIPS 

#Simulation dimensions 
Dimensions=(200,200)

#Initial composition (oil-component)
phi0=0.50

#Initial composition (solvent-component)
phis0=0.50

#Initialise the 'oil' order parameter field
Oil=Liquid(Dimensions,phi0)
 
#Initialise the 'solvent' order parameter field 
Solvent=solvent(Dimensions,phis0,Oil.dt)      

#Storage list of the system free energy
SystemEnergy=[]

#Add the initial energy of the system 
SystemEnergy.append(Oil.Calc_FreeEnergy(Oil.phi))

#Total number of simulation steps 
N=25000                     

#Start-point for determining computation time 
t0=process_time()  

for n in range(N+1):
    
    #Calculate the interaction parameter from the solvent field 
    Chi=Oil.Calc_Chi(Solvent.phi)
    
    #Arrests phase separation if threshold value of the IFT has been passed (i.e. nanoparticle adsorption)
    Oil.Arrest(Oil.phi)

    #Propagate the fields in time via the Cahn-Hilliard equation / Fick's second law
    Oil.Propagate(Oil.phi,Chi)
    Solvent.Propagate(Solvent.phi)
    
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