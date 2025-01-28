# -*- coding: utf-8 -*-
"""
Analysis of Bijel Formation Dynamics During Solvent Transfer-Induced Phase Separation Using Phase-Field Simulations
Pore size analysis / Critical IFT 
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

#%% Define functions that are used during pore-size determination 

#Function that extends the image with 0-values in both directions with a distance d 
def Extend_Image(I,d):
    Frame=np.zeros((I.shape[0]+2*d,I.shape[1]+2*d))
    Frame[d:Frame.shape[0]-d,d:Frame.shape[1]-d]=I
    return Frame

# Function that locates the oil-channels in a certain horizontal cross-section (C) of the image. Returns the total number of locate pores, their lenghts in the x-direction and the indices of their centres. 
def Find_Oil(C):
    
    #List with the sizes of the oil-channels in de x-direction
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


#%% Calculate the average pore size profiles of morphologies with different values of the cut-off interfacial tension 

#Create the framework of the resulting Figure
#Set fontsize
Fontsize=12

plt.figure()
plt.minorticks_on()
plt.xlabel('Depth ($\hat{y}/\hat{L}$)',fontsize=Fontsize)
plt.ylabel('Pore size ($\hat{d}/\hat{L}$)',fontsize=Fontsize)

plt.tick_params(axis='both',labelsize=Fontsize)

#Total number of images per batch 
Nt=20

#List with the different values of the cut-off IFT
sigma_c_list=(0.25,0.50,0.75,1,1.25,1.5,1.75)

for sigma_c in sigma_c_list:
    
    #List the contains the different morphologies for averaging 
    Profile_Storage=[]
    
    #Perform the pore size analysis for each image 
    for I in range(Nt):
        
        #Import the image (Enter path to folder with PoreSizeCalculation_SigmaC_Data in Path)
        Path=''
        Image=np.genfromtxt(Path+'\sigma='+str(sigma_c)+'\Morphology'+str(I)+'.txt')
    
        #Image dimensions 
        Ny=Image.shape[0]
        
        #Binarise the image
        Image[Image>=0.525]=1
        Image[Image<0.525]=0
        
        #Perform the actual image analysis 
        #Extend the image on all sides with 0-values. 
        E=10
        Image=Extend_Image(Image,E)
        
        # List with mean pore dimensions in the x-direction for each cross-section 
        PoreSize_dx=[]      
        # List with mean pore dimensions in the y-direction for each cross-section 
        PoreSize_dy=[]    
        
        #Set the range of horizontal cross-sections 
        dslice=2
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
        PoreSize=PoreSize[:int(len(Slices)*8/10)]
        Slices=Slices[:int(len(Slices)*8/10)]
        
        #Export the pore size profile to the storage list
        Profile_Storage.append(PoreSize)
        
    #Average the different morphologies for each batch 
    Average=np.zeros(len(Slices))
    for i in range(Nt):
        Average+=Profile_Storage[i]
    
    Average=Average/Nt
    
    #Visualise the average pore size profile for each best
    plt.plot(Slices,Average,label='$\sigma_c$='+str(sigma_c),marker='o',markersize=1.5)

plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.,fontsize=Fontsize)
