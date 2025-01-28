# -*- coding: utf-8 -*-
"""
Analysis of Bijel Formation Dynamics During Solvent Transfer-Induced Phase Separation Using Phase-Field Simulations
Pore size analysis / STrIPS bijels 
@author: J.M. Steenhoff
"""

#Import the required modules 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scienceplots

#Set Figure format 
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

#%% Define functions that are used during pore size determination 

#Function that extends the image with 0-values in both directions with a distance d 
def Extend_Image(I,d):
    Frame=np.zeros((I.shape[0]+2*d,I.shape[1]+2*d))
    Frame[d:Frame.shape[0]-d,d:Frame.shape[1]-d]=I
    return Frame

# Function that locates the oil-channels in a certain horizontal cross-section (C) of the image. Returns the total number of locate pores, their lenghts in the x-direction and the indices of their centres. 
def Find_Oil(C):
    
    #List with the px-sizes of the oil-channels in de x-direction
    PoreSize_x=[]
    #List with the indices of the (approximate) centre of each oil-channel
    Centre_x=[]
    
    #Counter for the size of the oil-channels
    p=0
    for i in range(len(C)):
        if C[i]==1:
            p+=1
            if C[i+1]==0:
                PoreSize_x.append(p)
                Centre_x.append(i-int(p/2))
                p=0
    return (len(PoreSize_x),PoreSize_x,Centre_x)

#Function that uses the list of x-coordinates of the oil-channel centres to find a corresponding pore-dimension in the y-direction. Als requires the y-value of the cross-section at which the channels were found.
def Calc_Oil_y(C,X,y):
    #List for the the pore-dimensions extending above the channel centre
    PoreSize_y1=[]
    #List for the the pore-dimensions extending below the channel centre
    PoreSize_y2=[]
    
    #Counter for the pore-dimension extending above the channel centre
    p_top=0
    #Counter for the pore-dimension extending below the channel centre. Starts at -1 to prevent double counting of the channel centre. 
    p_bottom=-1
    
    #For each located channel in a cross-section, determine the dimensions of the pore in the y-direction by summing the distances from the pore centre (x,y) to the nearest interface in opposite directions 
    for x in X:
        for i in range(y):
            if C[y-i,x]==1:
                p_top+=1
                if C[y-i-1,x]==0:
                    PoreSize_y1.append(p_top)
                    p_top=0
                    break    
        for i in range(C.shape[0]-y):
            if C[y+i,x]==1:
                p_bottom+=1
                if C[y+i+1,x]==0:    
                    PoreSize_y2.append(p_bottom)
                    p_bottom=-1
                    break
    
    PoreSize_y=np.array(PoreSize_y1)+np.array(PoreSize_y2)
    
    return (PoreSize_y,np.mean(PoreSize_y))

#%% Import the experimental data (confocal images)

#Paths for the .txt files of the confocal images with 0, 20 and 40 % solvent. 
Path0=''
Path20=''
Path40=''

#Import the images 
RawData0=np.genfromtxt(Path0)/255
RawData20=np.genfromtxt(Path20)/255
RawData40=np.genfromtxt(Path40)/255

RawData=(RawData0,RawData20,RawData40)

#Solvent fractions in the bore channel (solvent boundary condition)
Solvent=(0.00,0.20,0.40)

#Show the experimental confocal images (binarised)
for i in range(len(RawData)):
    plt.figure()
    plt.ylabel('Depth, $\hat{y}/\hat{L}$')
    plt.xlabel('Horizontal position, $\hat{x}/\hat{L}$')
    plt.title('$\phi_s^{BC}=$'+str('%.2f' % Solvent[i]))

    plt.imshow(RawData[i],MagentaBlack,extent=(0,1,1,0))
    
    Colorbar=plt.colorbar()
    Colorbar.set_label('$\phi$')

#%% Perform pore size analysis and plot the pore size profiles for different values of the solvent fraction in the bore channel (Experimental)

#Prepare the Figure frame 
plt.figure()
plt.minorticks_on()

plt.title('Experiment')
plt.xlabel('Depth, $\hat{y}/\hat{L}$')
plt.ylabel('Pore size, $\hat{d}/\hat{L}$')


#Different cut-off values for the different pore profiles 
C_list=[4,8,8]

#Storage list of the polynomial fits to the profiles 
Fits=[]

#Storage list for the used colours 
Colours=[]

for i in range(len(RawData)):
    
    Image=RawData[i]
    
    #Image dimensions 
    Ny=Image.shape[0]
    
    #Perform the actual image analysis 
    #Extend the image on all sides with 0-values. 
    E=10
    Image=Extend_Image(Image,E)
    
    # List with mean pore dimensions in the x-direction for each cross-section 
    PoreSize_dx=[]      
    # List with mean pore dimensions in the y-direction for each cross-section 
    PoreSize_dy=[]    
    
    #Set the range of horizontal cross-sections 
    dslice=4
    #Slices are chosen such that the regions of introduced 0-values due to image extension are not measured
    Slices=np.arange(E,Image.shape[0]-E,dslice)
    
    #Determine the pore dimensions in both x- and y-directions for each horizontal cross-section 
    for j in Slices:
        
        #Determines the number (N), x-dimensions (dx) and centre-coordinates (Xs) of the oil channels 
        N,dx,Xs=Find_Oil(Image[j])
        
        #If no oil-channels are found, dx (and therefore dy) are returned as 0 rather than NaN. 
        if np.isnan(np.mean(dx))==True:
            PoreSize_dx.append(0)
            PoreSize_dy.append(0)
        else:
            PoreSize_dx.append(np.mean(dx))
            PoreSize_dy.append(Calc_Oil_y(Image, Xs, j)[1])
    
    
    #Calculates the a measure for the pore size by averaging the determined dimensions in perpendicular directions 
    PoreSize=(np.array(PoreSize_dx)+np.array(PoreSize_dy))/2
    
    #Slices are re-scaled as to match the actual image dimensions, not the extended one
    Slices=Slices-E
    
    #Normalise pore sizes with respect to the system dimensions  
    PoreSize=PoreSize/Ny
    
    #Normalise the distance within the image with respect to the system dimensions  
    Slices=Slices/Ny
    
    #Cut the length of the profile
    C=C_list[i]
    PoreSize=PoreSize[C:int(len(Slices)-C)]
    Slices=Slices[C:int(len(Slices)-C)]
    
    #Plot the pore size profiles 
    plt.plot(Slices,PoreSize,marker='o',markersize=1.5,linestyle='',alpha=0.20)
    
    #Store the colour used for the plooted profile 
    Colours.append(plt.gca().get_lines()[i].get_color())
    
    #Fit the found profile against a 2nd order polynomial
    Fit=np.polynomial.Polynomial.fit(Slices,PoreSize,2)(np.arange(0,1,0.01))
    
    #Export the pore size fit to the storage list
    Fits.append(Fit)

#List of positions for the pore size maximum 
Max_pos=[]

#Plot the fits of the pore size profiles, extract the positions of the maximum pore size from the fits and plot the locations in the Figure 
for i in range(len(Fits)):
    Max_pos.append(np.arange(0,1,0.01)[np.argmax(Fits[i])])
    plt.plot(np.arange(0,1,0.01),Fits[i],color=Colours[i],label='$\phi_{s}^{BC}=$'+str(Solvent[i]))
    plt.plot(np.arange(0,1,0.01)[np.argmax(Fits[i])],np.max(Fits[i]),marker='*',color=Colours[i],zorder=2.5,markersize=7)

#%% Perform pore size analysis and plot the pore size profiles for different values of the solvent fraction in the bore channel (Simulation). Pore size analysis is performed on multiple (20) images for each solvent fraction. The average profile is subsequently used for fitting. 

#Create Figure framework 
plt.figure()
plt.minorticks_on()

plt.title('Simulation')
plt.xlabel('Depth, $\hat{y}/\hat{L}$')
plt.ylabel('Pore size, $\hat{d}/\hat{L}$')

#Total number of images per batch 
Nt=20

#List with the different values of the bottom boundary condition
#BC_list=(0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40) #Full list 
BC_list=(0,0.10,0.20,0.30,0.40)

#List the polynomial fits made with respect to the profiles 
Fits=[]

#List of set colours
Colours_set=['tab:blue','tab:red','tab:green','tab:purple','tab:orange']

for BC in range(len(BC_list)):
    
    #List the contains the different morphologies for averaging 
    Profile_Storage=[]
    
    #Perform the pore size analysis for each image 
    for I in range(Nt):
        
        #Import the image (in Path, fill in the path to the file containing the maps with different boundary condition (BC) values)
        Path=''
        Image=np.genfromtxt(Path+'\BC='+str(BC_list[BC])+'\Morphology'+str(I)+'.txt')
    
        #Image dimensions 
        Ny=Image.shape[0]
        
        #Binarise the image
        Image[Image>=0.5]=1
        Image[Image<0.5]=0
        
        #Perform the actual image analysis 
        #Extend the image on all sides with 0-values. 
        E=10
        Image=Extend_Image(Image,E)
        
        # List with mean pore dimensions in the x-direction for each cross-section 
        PoreSize_dx=[]      
        # List with mean pore dimensions in the y-direction for each cross-section 
        PoreSize_dy=[]    
        
        #Set the range of horizontal cross-sections 
        dslice=1
        #Slices are chosen such that the regions of introduced 0-values due to image extension are not measured
        Slices=np.arange(E,Image.shape[0]-E,dslice)
        
        #Determine the pore dimensions in both x- and y-directions for each horizontal cross-section 
        for i in Slices:
            
            #Determines the number (N), x-dimensions (dx) and centre-coordinates (Xs) of the oil channels 
            N,dx,Xs=Find_Oil(Image[i])
            
            #If no oil-channels are found, dx (and therefore dy) are returned as 0 rather than NaN. 
            if np.isnan(np.mean(dx))==True:
                PoreSize_dx.append(0)
                PoreSize_dy.append(0)
            else:
                PoreSize_dx.append(np.mean(dx))
                PoreSize_dy.append(Calc_Oil_y(Image, Xs, i)[1])
        
        
        #Calculates the a measure for the pore size by averaging the determined dimensions in perpendicular directions 
        PoreSize=(np.array(PoreSize_dx)+np.array(PoreSize_dy))/2
        
        #Slices are re-scaled as to match the actual image dimensions, not the extended one
        Slices=Slices-E
        
        #Normalise pore sizes with respect to the system dimensions  
        PoreSize=PoreSize/Ny
        
        #Normalise the distance within the image with respect to the system dimensions  
        Slices=Slices/Ny
        
        #Cut the length of the profile
        PoreSize=PoreSize[:int(len(Slices)*10/10)]
        Slices=Slices[:int(len(Slices)*10/10)]
        
        #Export the pore size profile to the storage list
        Profile_Storage.append(PoreSize)
         
    #Average the different morphologies for each batch 
    Average=np.zeros(len(Slices))
    for i in range(Nt):
        Average+=Profile_Storage[i]
    
    Average=Average/Nt
    
    #Fit the found profile against a 2nd order polynomial
    Fit=np.polynomial.Polynomial.fit(Slices,Average,2)(np.arange(0,1,0.01))
    
    #Export the pore size fits to the storage list
    Fits.append(Fit)

    #Visualise the average pore size profile for each BC
    plt.plot(Slices,Average,marker='o',markersize=1.5,linestyle='',alpha=0.20,color=Colours_set[BC])
    

#Plot the polynomial fits of the pore size profiles and extract maximum positions 

#List of positions for the pore size maximum 
Max_pos=[]

#Plot the fits of the pore size profiles, extract the positions of the maximum pore size from the fits and plot the locations in the Figure 
for i in range(len(Fits)):
    Max_pos.append(np.arange(0,1,0.01)[np.argmax(Fits[i])])
    plt.plot(np.arange(0,1,0.01),Fits[i],color=Colours_set[i],label='$\phi_{s}^{BC}=$'+str(BC_list[i]))
    plt.plot(np.arange(0,1,0.01)[np.argmax(Fits[i])],np.max(Fits[i]),marker='*',color=Colours_set[i],zorder=2.5,markersize=7)

plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
