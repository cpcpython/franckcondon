#!/usr/bin/env python

#%matplotlib ipympl

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

#---------------------------------------------------------------------
#  Simpson 1/3 rule for Normalizing Eigenfunctions
#---------------------------------------------------------------------
def Simpson(func,a,b,n):
# Simple Simpson Integrations over func array 
    n = n 
    h = (b - a) / (n - 1)
    I_simp = (h/3) * (func[0] + 2*sum(func[:n-2:2]) + 4*sum(func[1:n-1:2]) + func[n-1])
    return(I_simp)
# For ground state Wfn. normalization checking ... ######################################################
with open('1.41691PlotData.pkl', 'rb') as file:
# Load the object from the pickle file
    data = pickle.load(file) # ground
with open('2.00576PlotData.pkl','rb') as file:
    datae=pickle.load(file)   # excited

grid = 12000
FCgrid = np.zeros(shape=(11,11)) # FC visualize array

for i in range (len(data)): # upto range_32 is checked [ n=16 ] i upto 75 is checked.
    for j in range (len(datae)):
        ## FIND PSI BETWEEN[INEMIN TO INEMAX] INTERVAL
        yy =data[i][2] # shifted WfnG
        yye=datae[j][2] # shifted Wfn
        xx =data[i][1] # shifted x
        # Energi_i = (data[i][0][0]

        # Normalized Psi should be used from here onwards: yy = unNormalized Psi
        # little bit Normalization work; N = Sqrt[ Int[ f*f]dx] so,---------------------- Ground
        l = [x * x for x in yy] # l=Psi^2 here
        NormalizedC=math.sqrt(Simpson(l,xx[0],xx[grid-1],len(l)))
        # note we eliminated far right/left Psi values for noise reduction
        yyNormed = [x/NormalizedC for x in yy]

        # Normalized Psi should be used from here onwards: yy = unNormalized Psi
        # little bit Normalization work; N = Sqrt[ Int[ f*f]dx] so,---------------------- Excited
        l = [x * x for x in yye] # l=Psi^2 here
        NormalizedC=math.sqrt(Simpson(l,xx[0],xx[grid-1],len(l)))
        # note we eliminated far right/left Psi values for noise reduction
        yyeNormed = [x/NormalizedC for x in yye]

        # Checking whether Normalization correct or not:
        Psi2 = [x*x for x in yyNormed] # Psi^2
        IntPsiNorm=Simpson(Psi2,xx[0], xx[grid-1],(int)(grid))
#        print("Check: ∫(Psi_Normalized)^2 dx ==1 ? If  vg = ",i,"Calculated : \t",IntPsiNorm,"E_i = ",data[i][0][0])


    # for Excited state Normalization Checking ... #################################################

        # Checking whether Normalization correct or not:
        Psi2 = [x*x for x in yyeNormed] # Psi^2
        IntPsiNorm=Simpson(Psi2,xx[0], xx[grid-1],(int)(grid))
#        print("Check: ∫(Psi_Normalized)^2 dx ==1 ? If  ve = ",j,"Calculated : \t",IntPsiNorm,"E_i = ",datae[j][0][0])

        #Real FC Calculation is here
        Yge2 = list(map(lambda x, y: (x * y)**2, yyNormed, yyeNormed))
        FCfactor=Simpson(Yge2,xx[0],xx[grid-1],(int)(grid))
        # PE corrected vib level below
        print("FC FACTOR :: <ji> = < ",j,"|",i," >\t",FCfactor," for DEL_E [j-i]eV\t",27.211*((-0.5+datae[j][0][0])-(-1.0+data[i][0][0])))
        
        FCgrid[j][i]=FCfactor
# show the FC matrix FC_ij
import matplotlib.pyplot as plt
plt.imshow(FCgrid, interpolation='none')
plt.show()

