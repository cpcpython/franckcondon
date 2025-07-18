gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ 
gpkmohan_mbcet@general1gpk:~/Numerov/Morse/MorseV1-H2/ExtendR/nanohub/Github/RealTest$ cat ExcitedNumerov.v1.py 
#!/usr/bin/env python

#%matplotlib ipympl

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

#=======================================================================
# IMPORTANT PARAMETERS  : Excited state H2(+)
#=======================================================================

grid      = 12000        # N division from -xp to xp ie. grid number
global gridshift
gridShift = 4000         # need to cut end point's higher values; for lower vib.quantum numbers we need to increase it
ynm1      = 1e-16        # far left point value
yn        = 2e-16        # ynm1+1th point value "representably small number"
xp        = 8.0          # range of oscillator as [-xp...0...xp] in atomic unit
Egap      = 0.01         # an empirical one to fit WFN at energy levels
global node
node      = 0
Itermax   = 500000       # Iteration limit
EsearchMax =    0.       # for plotting, y's limit
Amp=1.0

global PlotData
PlotData = []

InEmin=0.1              # Starting E Minimum. it will automatically updated
InEmax=0.11             # Starting E Maximum, it will automatically updated
itern =10               # after this iteration program exit; if you want more levels increase it

global Xcomplete        # for complete range in x

print("Step size:h = ",xp/grid)  # it was 2*xp/grid since potential is symmetric wrt x=0 , now x=[0,xp]
#======================================================================

# define Morse
# calculate X range for Morse Function
Xcomplete = []
x=0
while x < xp:
        Xcomplete.append(x+2*xp/grid) # corresponding x values
        x=x+xp/grid

def Morse(x,D0,a0,Req0):
    return ( D0*(1-math.exp(-a0*(x - Req0)))**2 - D0)

#-------------------------------------------------------------
# main function to get value of wavefunction on the array/grid
#-------------------------------------------------------------
def callNumerov(E,xp,n,ynm1,yn):
    K =  0.0957091     # force constant From Mathematica Fit
    h = (xp)/n         # grid size from -xp to xp
    m = 1837*1/2       # reduced mass of Hydrogen molecule in atomic unit
# starting points
    y_n_minus_1= ynm1 # first y
    y_n        = yn   # second y
    y          =[]    # third y, the calculated point
    xarray     =[]    # x values in f(x
    x=.0 # before, for SHO only x=-xp             # x is a starting point, say from the far left point
    while x < xp:     # Numerov method loop
        g_n_minus_1    =  2*m*(E-Morse(x,D0,a0,Req0))
        g_n            =  2*m*(E-Morse(x+h,D0,a0,Req0))
        g_n_plus_1     =  2*m*(E-Morse(x+2*h,D0,a0,Req0))
        y_n_plus_1     = (2*y_n*(1 - (5*h**2/12)*g_n) - y_n_minus_1*(1 + (h**2/12)*g_n_minus_1)) / (1 + (h**2/12)*g_n_plus_1)
        y.append(y_n_plus_1) # Solution array, unnormalized
        xarray.append(x+2*h) # corresponding x values
        x=x+h
        y_n_minus_1=y_n
        y_n        =y_n_plus_1

    return xarray,y

#---------------------------------------------------------------------
#  Simpson 1/3 rule for Normalizing Eigenfunctions
#---------------------------------------------------------------------
def Simpson(func,a,b,n):
# Simple Simpson Integrations over func array 
    n = n 
    h = (b - a) / (n - 1)
    I_simp = (h/3) * (func[0] + 2*sum(func[:n-2:2]) + 4*sum(func[1:n-1:2]) + func[n-1])
    return(I_simp)


#-----------------------------------------------------------------------
# INITIAL BRACKETING OF ENERGY [Emin < delE < Emax] where delE is the Energy and we find Wavefunction of this delE
#-----------------------------------------------------------------------
def findHermiteFunctions(Emin, Emax):
    # gridshift adjustement for higher quantum numbers
    if(Emin > -0.05):
        global gridShift 
        gridShift=2000
    if(Emin > -0.02):
        global grishShift
        gridShift=1500

    InEmin=Emin; InEmax=Emax # Just storing the original values


    delE=Emin; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymin=yo[(int)(grid)-1]

    delE=Emax; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
# error
    Eendymax=yo[(int)(grid)-1]

    Emid=((Emin+Emax)/2.);delE=Emid ; xo,yo=callNumerov(delE,xp,grid,ynm1,yn)
    Eendymid=yo[(int)(grid)-1]

    # BISECTION TO FIND APPROX. LOCATION OF delE in which y(+xp) ~ 0
    bisecN=50
    print("================ <<  FIRST BISECTION BEGIN >> ================")
    for egy in range(1,bisecN,1):

        delEmin=Emin; xo,yo=callNumerov(delEmin,xp,grid,ynm1,yn); Eendymin=yo[(int)(grid)-1]
        delEmax=Emax; xo,yo=callNumerov(delEmax,xp,grid,ynm1,yn); Eendymax=yo[(int)(grid)-1]
        Emid=((Emin+Emax)/2.);
        delEmid=Emid ; xo,yo=callNumerov(delEmid,xp,grid,ynm1,yn); Eendymid=yo[(int)(grid)-1]
        print("First bisection - Loop-N, delEmin, delEmid, delEmax::\t", egy,delEmin,delEmid,delEmax)

        # if it finds ideal biseaction region which contains a definte Root, it will break the loop over here
        if(Eendymid*Eendymin < 0 ):
            print("Exit in First bisection - Values of delEmin,delEmid,delEmax::\t", delEmin,delEmid,delEmax)
            print("Ideal bisective region found ...")
            break;
        if(Eendymid*Eendymax < 0 ): # ideal bracket bw. mid-mx
            print("Exit in First bisection - Values of delEmin,delEmid,delEmax::\t", delEmin,delEmid,delEmax)
            print("Ideal bisective region found ...")
            break;
# Bisection method - Two possibilities [1] monotonically decreasing but all positive or negative values for Emid,Emin;
# or [2] region with real root. For Possibility [1] Lowest decreasing part is found for root search, if f(x) are having same sign
        if(abs(Eendymid) < abs(Eendymin)):
            Emin=delEmid;

    print("================ <<< FIRST BISECTION END >>> ================")

    # ------------------------------------------------------------------------------------
    # Exiting if Emin and Emax didnt change at all 
    # sometime it wont give sufficient result, so it can exited from the below for loop
    # egy != 1 need since sometime a single bisection loop find optimum bisection bracket
    if((Emin==InEmin) and (Emax==InEmax) and egy != 1):
        print("No solutions in this Energy Interval,[",Emin,",",Emax, "]. Exiting ...")
        return 0,xo[0:grid-gridShift],yo[0:grid-gridShift]
    # -------------------------------------------------------------------------------------
    # important: If first section gives Emin~Emid~Emax we dont go furthur and should be returned Null
    tolSec=1e-12
    if(abs(Emid-Emin)<tolSec and abs(Emid-Emax)< tolSec and abs(Emin-Emax)<tolSec):
        print("No solutions in this Energy Interval where Emin~Emid~Emax: Exiting from the Second Bisection ...")
        return 0,xo[0:grid-gridShift],yo[0:grid-gridShift] # exiting ...

    bs1=Emin;bs2=Emid;bs3=Emax
    print("*** *** *** <<< SECOND BISECTION BEGIN >>> *** *** *** ",Emin,Emid,Emax)
    secondbs=1; tolSec=0.0000001    # Second bisection for finding approximated Eigenfunction
    while(secondbs < 50):          # Hopefully 50 bisection is enough !

        delEmid=(delEmax+delEmin)/2
        if((Eendymin)*(Eendymid) < 0):
            Emax=delEmid; print("Root=================1 N Emin Emax:\t",secondbs, Emin,Emax)
        if ((Eendymax)*(Eendymid) <0 ):
            Emin=delEmid; print("Root=================2 N Emin Emax:\t",secondbs, Emin,Emax)

        delEmin=Emin; xo,yo=callNumerov(delEmin,xp,grid,ynm1,yn); Eendymin=yo[(int)(grid)-1]
        delEmax=Emax; xo,yo=callNumerov(delEmax,xp,grid,ynm1,yn); Eendymax=yo[(int)(grid)-1]
        Emid=((Emin+Emax)/2.);
        delEmid=Emid ; xo,yo=callNumerov(delEmid,xp,grid,ynm1,yn); Eendymid=yo[(int)(grid)-1]

        # the below means points doesnt changeing att all in this loop
        if(secondbs>10000):
            if(abs(bs1-Emin)<tolSec and abs(bs2-Emid)< tolSec and abs(bs3-Emax)<tolSec):
                print("No solutions in this Energy Interval,[",Emin,",",Emax, "] Exiting from the Second Bisection ...")
                return 0,xo[0:grid-gridShift],yo[0:grid-gridShift]; # exiting ...

        secondbs=secondbs+1
        if(abs(Emid-InEmax) <=2e-16):# sometime Emid tends to the limit of InEmax, that give error, so it should be avoided
            print("Emid-InEmax are very small, Exiting....")    
            return 0,xo[0:grid-gridShift],yo[0:grid-gridShift]

        if(abs(Emid-Emin)< 2e-16): # Crucial step ~ Machine Precision
            print("*** Break in Second Bisection ***")
            # Right-End Points contains noises so it should be "symmetrically" removed like xo[100:1500] yo[100:1500]
# if we want to save PNG files 
            plt.plot(xo[0:grid-gridShift],yo[0:grid-gridShift], linestyle='--', marker='o', color='g')
            plt.savefig('LARGE_Emidsho_'+str(Emid)+'H2.png')
            # PlotData contains the Full gridso Padding of zeros applied only in yy (due to its deviated values at ends)
            zeroP = [0] * gridShift
            PlotData.append([[Emid],xo[0:grid],yo[0:grid-gridShift]+zeroP ])
            plt.close()
            print("*************  Convergence Achieved ************")

            # save this FULL grid based x,Wavefunction(un normalized) for FC factor info.
            with open(str(Req0)+"PlotData.pkl", "wb") as file:
                pickle.dump(PlotData, file)

            # Main Return
            global node
            node=node+1
            return Emid,xo[0:grid-gridShift],yo[0:grid-gridShift]#break
        
    # Main Results
    return Emid,xo[0:grid-gridShift],yo[0:grid-gridShift] # Caution: <<<<< No effect ????? seems its not reaching !!!
#-------------------------------------------------------------------------------------
# Function to plot Potential Parabola and Classical Turn points etc.
#-------------------------------------------------------------------------------------
def PlotParabolaPlus(Energies,K,XAll,PsiAll):
    #---------------------------------------
    # Plot (1/2)Kx**2 Parabola
    #---------------------------------------
    y=[]
    xarray=[]
    y=[ Morse(x,D0,a0,Req0) for x in Xcomplete]
    xarray=Xcomplete
    
    #-----------------------------------------
    En=Energies[0]

    # --------------------------------------- Wavefunction Plot -----------------------------------
    # insert Lines of Energy
    fig, ax = plt.subplots()
    ax.plot(xarray, y, linewidth=2.0) # plot parabola

    ax.set_xlabel('Morse Potential in atomic unit', fontweight ='bold')
    ax.set_ylabel('Energy/Wfn (arb. unit)', fontweight ='bold')
    for i in range(len(Energies)):
        print("Energy Level,n=",i+1," E = ", Energies[i])
        En=Energies[i]
        ax.hlines(y=En, xmin=0, xmax=xp, linewidth=1, color='y') #0 xp
        mxPsi=max(PsiAll[i]); mnPsi=min(PsiAll[i])
        Scale=mxPsi-mnPsi
        #Scale=1
        print("-----------------------------------------------------------------------")
        print(mxPsi,mnPsi,Scale)
        print("------------------------------------------------------------------------")
        EadjustedWfn=list(map(lambda x:En+Amp*Egap*(x/Scale),PsiAll[i]) ) # note x/Scale : Confined Psi in Scale
        ax.plot(Xcomplete[0:len(EadjustedWfn)] ,EadjustedWfn)
    
    pickle.dump(fig, open('FigureWfn.fig.pickle', 'wb'))              # saving interactive plot - wfn
    plt.savefig('P1DWavefunction.png')
    plt.show()
    plt.close()

    # --------------------------------------- Probability Plot -----------------------------------
    # insert Lines of Energy
    fig1, ax1 = plt.subplots()
    ax1.plot(xarray , y, linewidth=2.0) # plot parabola

    ax1.set_xlabel('1D Box length in atomic unit', fontweight ='bold')
    ax1.set_ylabel('Energy/Probability (arb. unit)', fontweight ='bold')
#   In first we make Psi*Psi2
    PsiAll2  = [[item**2 for item in sublist] for sublist in PsiAll]
#    PsiAll2  = np.square(PsiAll) #list(map(lambda x: x ** 2, PsiAll)) #
    for i in range(len(Energies)):
        En=Energies[i]
        ax1.hlines(y=En, xmin=0, xmax=xp, linewidth=1, color='y')
        mxPsi=max(PsiAll2[i]); mnPsi=min(PsiAll2[i])
        Scale=mxPsi-mnPsi
#        Scale=1
        print("-----------------------------------------------------------------------")
        print(mxPsi,mnPsi,Scale)
        print("------------------------------------------------------------------------")

        EadjustedProb=list(map(lambda x:En+Amp*Egap*(x/Scale),PsiAll2[i])) # note x/Scale : Confined Psi^2 in Scale
        ax1.plot(Xcomplete[0:len(EadjustedProb)],EadjustedProb)


    pickle.dump(fig1, open('FigureProb.fig.pickle', 'wb'))
    plt.savefig('P1DProbability.png')
    plt.show()
    plt.close()

#-------------------------------------------------------------------------------------
# Main Starting Input of Energies
# It first scans Energy from InEmin to InEmax then try to go up
#-------------------------------------------------------------------------------------

def ScanHermitefunctions(InEmin,InEmax,itern,nodeMax):
    #InEmin=0.00 ; InEmax = 0.01
    #Converged Energies, stores here
    Econverged=[]
    PsiAll=[]
    XAll=[]
    
#    if(InEmin > -0.1): # for higher v we need lesser gridshift
#        global gridShift
#        gridShift = 3000

    for i in range (itern): # upto range_32 is checked [ n=16 ] i upto 75 is checked.
        print("\n\n Searching Eigenfunction in [InEmin,InEmax] : " , InEmin,InEmax)
        ## FIND PSI BETWEEN[INEMIN TO INEMAX] INTERVAL ; Main Program Call Begins ...
        ee,xx,yy = findHermiteFunctions(InEmin,InEmax)
        if(ee != 0):
            Econverged.append(ee)
            # Normalized Psi should be used from here onwards: yy = unNormalized Psi
            # little bit Normalization work; N = Sqrt[ Int[ f*f]dx] so,----------------------
            l = [x * x for x in yy] # l=Psi^2 here
            NormalizedC=math.sqrt(Simpson(l,xx[0],xx[grid-gridShift-1],len(l)))
            #NormalizedC=math.sqrt(Simpson(l,xx[0],xx[(int)(grid/2)-1],len(l)))
            # note we eliminated far right/left Psi values for noise reduction

            yyNormed = [x/NormalizedC for x in yy]
            #--------------------------------------------------------------------------------
            # Choice - 1
            # PsiAll.append(yy)       # for UnNormalized Psi :
            # Choice - 2
            PsiAll.append(yyNormed) # if we want Normalized Psi
            #--------------------------------------------------------------------------------
            # Checking whether Normalization correct or not:
            Psi2 = [x*x for x in yyNormed] # Psi^2
            IntPsiNorm=Simpson(Psi2,xx[0], xx[grid-gridShift-1],(int)(grid-gridShift-1))
            print("Normalization check: ∫(Psi_Normalized)^2 dx ==1 ? If  n= \t",i,"Calculated Value: \t",IntPsiNorm,"")

            XAll=xx
        InEmin=InEmax
        InEmax=InEmax+0.005  # for Rydberg levels this 0.005 should be decreased

        if(InEmax >EsearchMax or node > nodeMax ): # means Node > 10 wfn dont need to be calculated
            break  # to stop loop

    return Econverged,XAll,PsiAll


# **Acual Calculation Section**
# By giving a suitable numerical value of xp (-xp to xp consists the bond elongation/compression with respect to the equilibrium bond length, BL, See the Paramter section), and other important parameters, now one can start the calculations. Note that the Probaility and Wavefunction plot are 'scaled' to fit into the parabola, legibly.

# In[3]:


K= 0.0957091                # force constant of H2 molecule
EsearchMax =    -0.005      # for plotting, y's limit
InEmin     =    -0.18       # Starting E Minimum. it will automatically updated
InEmax     =    -0.15       # Starting E Maximum, it will automatically updated
itern      =     75         # after this iteration program exit; if you want more levels increase it
nodeMax    =     10         # Limit the node of wfn to stop the run
# Ground state Morse Parameters
D0   = 0.102928 
a0   = 0.681859  
Req0 = 2.00576

Econverged,XAll,PsiAll=ScanHermitefunctions(InEmin,InEmax,itern,nodeMax)
PlotParabolaPlus(Econverged,K,XAll,PsiAll)

# Replotting
#figxw = pickle.load(open('FigureWfn.fig.pickle', 'rb'))  # Show the Wavefunction Figure interactively
#figxp = pickle.load(open('FigureProb.fig.pickle', 'rb')) # Show the Probability Figure interactively

# saving Distance X PsiNormalized
np.save('array_R', Xcomplete)


#with open(str(Req0)+"PlotData.pkl", "wb") as file:
#    pickle.dump(PlotData, file)

#np.save('array_PsiNorm', PsiAll)

