#%%
import numpy as np
import time
import pandas as pd
import csv
from numba import jit
import spglib



# for creating a responsive plot
#matplotlib widget
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from fractions import Fraction
import muesr
from muesr.core.atoms import Atoms
from muesr.core import Sample           # The object that contains the information
from muesr.engines.clfc import locfield # Does the sum and returns the results
from muesr.engines.clfc import find_largest_sphere # A sphere centered at the muon is the correct summation domain
from muesr.i_o import load_cif          # To load crystal structure information from a cif file
from muesr.utilities import mago_add, show_structure # To define the magnetic structure and show it
from muesr.utilities import muon_find_equiv
np.set_printoptions(suppress=True,precision=4)       # to set displayed decimals in results
from scipy.signal import find_peaks


#%%
def setStructure(q,i_q,single_mm,cif_location,supercell):
    
    #create sample object
    s = Sample()
    # load lattice structure from *.cif file
    load_cif(s, cif_location)
    all_mm = np.zeros((len(q),len(single_mm),3),dtype=complex)
    for i in range(len(q)):
        # define copies of the magnetic structure mm which will be modified to fit the right ordered wave vector q
        all_mm[i] = list(mm)
        # use the index i_q to set to zero the right mm
        if len(i_q) != 0:
            all_mm[i,i_q[i]] = [0,0,0]
        print(all_mm[i])
        # initialize new magnetic structure
        s.new_mm()
        # add the right ordered wave vector to new structure
        s.mm.k = q[i]
        # add the right magnetic structure
        s.mm.fc = all_mm[i]
    return s
def field_at_site(muon,commensurate_type,nangles):
    muesr.utilities.muon.muon_reset(s)
    s.add_muon(muon)
    total_dip = np.zeros_like([0.,0.,0.],shape=[len(s.muons),3])
    for i in range(s.mm_count):
        s.current_mm_idx = i
        radius=find_largest_sphere(s,supercell)
        if commensurate_type == "s":
            r = locfield(s,commensurate_type,supercell,radius)
            B_dip=np.zeros([len(s.muons),3])
            for i in range(len(s.muons)):
                B_dip[i]=r[i].D*10000
            total_dip += B_dip

        if commensurate_type == "i":
            s.current_mm_idx = 0;      # N.B.: indexes start from 0 but idx=0 is the transverse field!
            r_RH = locfield(s, 'i',[50,50,50],100,nnn=3,nangles=nangles)
            N_BINS=1000
            RH_Hist=np.zeros(N_BINS)
            bin_range=np.zeros(N_BINS+1)
            for i in range(len(r_RH)):
                hist, bin_range = np.histogram(np.linalg.norm(r_RH[i].T*10000, axis=1), bins=N_BINS, range=(0,500))
                RH_Hist += hist
            # just for plotting, gnerate intermediate positions for bins of the histogram
            mid_of_bin = bin_range[0:-1]+0.5*np.diff(bin_range)
            # find peaks of histogram

            peaks_RH, _ = find_peaks(RH_Hist)

            if len(mid_of_bin[peaks_RH]) == 0:
                total_dip += 0
            total_dip += max(mid_of_bin[peaks_RH])





            """
            r = locfield(s,commensurate_type,supercell,radius,nangles=nangles)
            sum_angles = [0,0,0]
            for i in range(nangles):
                if not np.isnan(r[0].T[i][0]) and not np.isnan(r[0].T[i][1]) and not np.isnan(r[0].T[i][2]):
                    sum_angles += r[0].T[i]
                    #sum_angles[i] = [r[0].T[i][0]/nangles,r[0].T[i][1]/nangles,r[0].T[i][2]]/nangles # we want the mean of each component x y z
            sum_angles[0] = sum_angles[0]/nangles
            sum_angles[1] = sum_angles[1]/nangles
            sum_angles[2] = sum_angles[2]/nangles
            total_dip = sum_angles
                # returns 36000 x B, 36000 y B, 36000 z B
                # we want norm of x, norm of y, norm of z
            """

    

    # see https://stackoverflow.com/questions/38698277/plot-normal-distribution-in-3d
    norm1 = np.round(np.linalg.norm(total_dip[0]))
    if len(total_dip) != 1 :
        norm2 = np.round(np.linalg.norm(total_dip[1]))
        if norm1 <= norm2:
            highest = norm2
        else: highest = norm1
    else: highest = norm1

    return highest

# removes empty values from X, Y, Z and F
# we don't want to show the empty values so we append all non-empty values to new temporary array
def removeEmpty(x,y,z,f):
    newX=[]
    newY=[]
    newZ=[]
    newF=[]
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i,j])):
                if x[i,j,k] is not None:
                    newF=np.append(newF,f[i,j,k])
                    newX=np.append(newX,x[i,j,k])
                    newY=np.append(newY,y[i,j,k])
                    newZ=np.append(newZ,z[i,j,k])    
    return newX,newY,newZ,newF
# iterate over precision x precision x precision matrix, total volume 1 in normalized lengths
# time taken scales in 0.5 sec * n^3
def getCubeFields(n,cells,commensurate_type,nangles,timestamp=True):
    # starting point of 0 is assumed
    xlim = cells[0]*n+1
    ylim = cells[1]*n+1
    zlim = cells[2]*n+1 # we wish to scan 6 unit cells in the z direction
    
    start = time.perf_counter()
    now = start

    x = np.full(shape=(xlim,ylim,zlim),fill_value=None)
    y = np.full(shape=(xlim,ylim,zlim),fill_value=None)
    z = np.full(shape=(xlim,ylim,zlim),fill_value=None)
    f = np.full(shape=(xlim,ylim,zlim),fill_value=None)
    
    
    print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(now-start)),", % completion:",0)#+j/(int(n)**2)))
    for k in range(0,zlim): # c direction
        for j in range(0,ylim): # b direction
            for i in range(0,xlim): # a direction
                muon = [i/(n),j/(n),k/(n)]
                #muon = [(i-(1/n))/(n-1),(j-(1/n))/(n-1),(k-(1/n))/(n-1)]
                x[i,j,k] = muon[0]
                y[i,j,k] = muon[1]
                z[i,j,k] = muon[2]
                f[i,j,k] = field_at_site(muon,commensurate_type,nangles)
                if np.isnan(f[i,j,k]):
                    f[i,j,k] = 10**100
        
        if timestamp == True:
            now = time.perf_counter()
            completion = ((k+1)/int(zlim))
            elapsed = now - start
            estimated_total = elapsed / completion
            estimated_time_remaining = estimated_total - elapsed
            print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(elapsed)),", % completion:",np.round(100*completion,5),\
                  "Estimated time remaining:",time.strftime("%H:%M:%S", time.gmtime(estimated_time_remaining)))#+j/(int(n)**2)))
    stop = time.perf_counter()
    if timestamp == True:
        duration = "Total duration : "+str(time.strftime("%H:%M:%S", time.gmtime(stop-start)))+"\n"
        header = "Unit cell divided in "+str(n)+" parts\n"
        x,y,z,f = removeEmpty(x,y,z,f)
        return x,y,z,f,[duration,header]
    else: return x,y,z,f


def saveFile(filename,x,y,z,f,header=""):
    # write / text as opposed to read / binary ("rb")
    with open(filename,"wt") as fp:
        #writer = csv.writer(fp)
        
        write = csv.writer(fp)
        write.writerow(header)
        for i in range(len(x)):
            write.writerow([x[i],y[i],z[i],f[i]])

# open save file instead of recalculating
# option to create temporary files to avoid overwriting
def openSingleFile(filename):
    with open(filename, newline='') as csvfile:
        header = csv.reader(csvfile)
        header1 = next(header,None) # read line 1
        header2 = next(header,None) # read line 2
        reader = csv.reader(csvfile)
        
        data = [list(c) for c in zip(*reader)]
        
        
    x=np.array(list(map(float, data[0])))
    y=np.array(list(map(float, data[1])))
    z=np.array(list(map(float, data[2])))
    f=np.array(list(map(float, data[3])))
    headers = header1 + header2
    return x,y,z,f,headers

def getAtomPositions(atom_positions,cells=[1,1,1]):    
    # atom_positions is shaped (#atoms, #properties)
    s_x,s_y,s_z,s_name,s_nth = [],[],[],[],[]
    for i in range(len(atom_positions[:,0])):     # number of atoms
        # loop over all duplicate atoms in cells in x
        for j in range(cells[0]+1):
            # loop over all duplicate atoms in cells in y
            for k in range(cells[1]+1):
                # loop over all duplicate atoms in cells in z
                for l in range(cells[2]+1):
                    x = float(atom_positions[i,0]) + j
                    y = float(atom_positions[i,1]) + k
                    z = float(atom_positions[i,2]) + l
                    name = atom_positions[i,3]
                    nth = int(atom_positions[i,4])
                    s_x=np.append(s_x,x)
                    s_y=np.append(s_y,y)
                    s_z=np.append(s_z,z)
                    s_name=np.append(s_name,name)
                    s_nth=np.append(s_nth,nth)
    
    
    properties_elements_list = s_x,s_y,s_z,s_name,s_nth 
    return properties_elements_list

# electron density obtained from DFT calculation
def DFTplot(filename,dim,interval0,interval1):
    filename = "C:/Users/User/OneDrive - Universite de Montreal/Masters/Codes/MuESR_Outputs/density_Ce3PtIn11.txt"
    length = dim[0]*dim[1]*dim[2]
    X = np.full(length,fill_value=None)
    Y = np.full(length,fill_value=None)
    Z = np.full(length,fill_value=None)
    n=0
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                X[n] = k/dim[2]
                Y[n] = j/dim[1]
                Z[n] = i/dim[0]
                n = n+1
    
    frame = pd.read_table(filename, sep='\s+')
    frame = frame.to_numpy()
    F = np.ravel(frame)[:-1]
    # assign x,y,z values ordered in the same way as the .xsf file
    # all 73 by 251 cells with x = 1/73*a, first with y = 1/73*b and z going through all /251*c possibilities
    # all 73 by 251 cells with x = 2/73*a
    # electron density doesnt go higher than 3.7
    chosenX,chosenY,chosenZ,chosenF = chooseInterval(X,Y,Z,F,interval0,interval1)
    
    return chosenX,chosenY,chosenZ,chosenF

def plot3d(X,Y,Z,F,a,b,c,cells=[1,1,1],normalized=True,atom_positions=False,filenameDFT = False,dimDFT = False,title=False):
    # creating 3d figures
    fig = plt.figure(figsize=(6, 6),layout="tight")
    ax = fig.add_subplot(111, projection='3d')
    # configuring colorbar
    color_map = cm.ScalarMappable(cmap=cm.cool)
    color_map.set_array(F)
    if len(atom_positions) != 1:
        s_x,s_y,s_z,s_name,s_nth = getAtomPositions(atom_positions,cells)
        print(len(s_x))
        for i in range(len(s_x)): #plot each point + it's index as text above
            if s_name[i] == "Ce":
                s_color = "dimgray"
                if s_nth[i] == 2:
                    s_color = "black"
            if s_name[i] == "In":
                s_color = "red"
                if s_nth[i] == 2:
                    s_color = "pink"
                if s_nth[i] == 3:
                    s_color = "green"
                if s_nth[i] == 4:
                    s_color = "royalblue"
            if s_name[i] != "Ce" and s_name[i] != "In":
                s_color = "orange"
            if (s_x[i] > cells[0] or s_y[i] > cells[1] or s_z[i] > cells[2]):
                continue
            atom_size = 120/(sum(cells)/3)
            if normalized == False:
                ax.scatter(0.5*a,0.5*b,0.5*c,color="gold",label="70% muon site",s=atom_size,clip_on=False)
                ax.scatter(0.0,0.5*b,0.27755*b,color="lime",label="30% muon site",s=atom_size,clip_on=False)
                ax.scatter(0.5*a,0.0,0.27755*b,color="lime",label="30% muon site",s=atom_size,clip_on=False)             
                ax.scatter(s_x[i]*a,s_y[i]*b,s_z[i]*c,color=s_color,label=s_name[i]+"("+str(s_nth[i])+")",s=atom_size,clip_on=False)

            if normalized == True:
                ax.scatter(0.5,0.5,0.5,color="gold",label="70% muon site",s=atom_size,clip_on=False)
                ax.scatter(0.0,0.5,0.27755,color="lime",label="30% muon site",s=atom_size,clip_on=False)
                ax.scatter(0.5,0.0,0.27755,color="lime",label="30% muon site",s=atom_size,clip_on=False)
                ax.scatter(s_x[i],s_y[i],s_z[i],color=s_color,label=s_name[i]+"("+str(s_nth[i])+")",s=atom_size,clip_on=False)


    # creating the heatmap

    if normalized == False: 
        img = ax.scatter(X*a,Y*b,Z*c,c=cm.cool(F/max(F)), marker='o',label="Muon sites",s=10,clip_on=False)
    if normalized == True: 
        img = ax.scatter(X,Y,Z,c=cm.cool(F/max(F)), marker='o',label="Muon sites",s=10,clip_on=False)
        
    
    if filenameDFT != False:
        Xden,Yden,Zden,Fden = DFTplot(filenameDFT,dimDFT,3.35000000001,4)
        return Xden,Yden,Zden,Fden
        if normalized == False:
            ax.scatter(Xden*a,Yden*b,Zden*c,c=cm.Wistia(Fden),label = "Electron density",s=20,clip_on=False,alpha = 0.8)
        if normalized == True:
            ax.scatter(Xden,Yden,Zden,c=cm.Wistia(Fden), marker = 'o',label = "Electron density",s=20,clip_on=False, alpha = 0.8)
        #Xden,Yden,Zden,Fden = DFTplot(filenameDFT,dimDFT,3.1000000001,3.5)
        #if normalized == False:
        #    ax.scatter(Xden*a,Yden*b,Zden*c,c=cm.Wistia(Fden),label = "Electron density",s=20,clip_on=False,alpha = 0.2)
        #if normalized == True:
        #    ax.scatter(Xden,Yden,Zden,c=cm.Wistia(Fden), marker = 'o',label = "Electron density",s=20,clip_on=False, alpha = 0.2)
        #Xden,Yden,Zden,Fden = DFTplot(filenameDFT,dimDFT,2.9,3.1)
        #if normalized == False:
        #    ax.scatter(Xden*a,Yden*b,Zden*c,c=cm.Wistia(Fden),label = "Electron density",s=20,clip_on=False,alpha = 0.02)
        #if normalized == True:
        #    ax.scatter(Xden,Yden,Zden,c=cm.Wistia(Fden), marker = 'o',label = "Electron density",s=20,clip_on=False, alpha = 0.02)


        
        
    if title != False:
        # adding title and labels
        ax.set_title(title)


    if normalized == False: 
        ax.set_xlabel('X (Å)',labelpad=6)
        ax.set_ylabel('Y (Å)',labelpad=6)
        ax.set_zlabel('Z (Å)',labelpad=6)
        ax.set_xlim([0, cells[0]*a])
        ax.set_ylim([0, cells[1]*b])
        ax.set_zlim([0,cells[2]*c])
    if normalized == True: 
        ax.set_xlabel('X',labelpad=6)
        ax.set_ylabel('Y',labelpad=6)
        ax.set_zlabel('Z',labelpad=6)
        ax.set_xlim([0, cells[0]])
        ax.set_ylim([0, cells[1]])
        ax.set_zlim([0,cells[2]])

    #plt.subplots_adjust(wspace=0.5, hspace=0.5)
    ax.set_box_aspect([a,b,c])
    ax.tick_params(axis='both', pad=15)

    plt.xticks(color='w',alpha=0)

    # yticks color white
    plt.yticks(color='w',alpha=0)

    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique),loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=8)

    legend_without_duplicate_labels(ax)

    # displaying plot

    plt.show()
    
def chooseInterval(x,y,z,f,minF,maxF,show=False):
    # return index of all F with value 116 +- 4
    chosenI = [] # chosen index
    chosenX = [] # value at chosen index
    chosenY = []
    chosenZ = []
    chosenF = []
    for i, val in enumerate(f):
        if val >= minF and val <= maxF:
            chosenI = np.append(chosenI,i)
            chosenX  = np.append(chosenX,x[i])
            chosenY  = np.append(chosenY,y[i])
            chosenZ  = np.append(chosenZ,z[i])
            chosenF  = np.append(chosenF,f[i])
            if show == True:
                print("X:",x[i],"Y:",y[i],"Z:",z[i],"Index:",i,"Value of Field (G):",f[i])
    return chosenX,chosenY,chosenZ,chosenF

def combineDataFiles(files):
    x=[]
    y=[]
    z=[]
    f=[]
    for i in filenames:
        filename = i
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [list(c) for c in zip(*reader)]
    x=np.append(x,list(map(float, data[0])))
    y=np.append(y,list(map(float, data[1])))
    z=np.append(z,list(map(float, data[2])))
    f=np.append(f,list(map(float, data[3])))
    return x,y,z,f
#%%
# MAIN
plt.close("all")
a=4.6874
b=4.6874
c=16.891
n_atoms = 15 # number of atoms in molecule
n_properties = 5
# X, Y, Z, atom name, nth (5 properties)
atoms = np.zeros(shape=(n_atoms,n_properties),dtype=object)
atoms[0]=[0,0.5,0.41167,"In",1]
atoms[1]=[0,0.5,0.5883,"In",1]
atoms[2]=[0.5,0,0.5883,"In",1]
atoms[3]=[0.5,0,0.4117,"In",1]
atoms[4]=[0.5,0.5,0.27784,"In",2]
atoms[5]=[0.5,0.5,0.7222,"In",2]
atoms[6]=[0,0.5,0.13792,"In",3]
atoms[7]=[0,0.5,0.8621,"In",3]
atoms[8]=[0.5,0,0.8621,"In",3]
atoms[9]=[0.5,0,0.1379,"In",3]
atoms[10]=[0.5,0.5,0,"In",4]
atoms[11]=[0,0,0.5,"Pt",1]
atoms[12]=[0,0,0.27755,"Ce",1]
atoms[13]=[0,0,0.7225,"Ce",1]
atoms[14]=[0,0,0,"Ce",2]

mm = np.zeros((15,3),dtype=complex)
mc = 2.54 # muB for electronic magnetic moment
mm[10],mm[11]=[0,0,mc/40+0.j],[0,0,mc/40+0.j] # both Ce(1) = (0,0,mc)
mm[14]=[0,0,mc+0.j]                           # Ce(2) = (0,0,mc/40)

q1 = [1/2,1/2,1/6]         # a
#q1 = [1/2,1/2,1/3]         # b
#q2 = [1/2,1/2,1/2]         # a
q2 = [1/2,1/2,0.0]         # b

all_q = [q1,q2] # modify this to change a <--> b
# index of magnetic structure mm to be changed to [0,0,0] for each ordered wave vector q
i_q = [[14],[10,11]]
cif_location = "C:/Users/User/OneDrive - Universite de Montreal/Masters/Laboratoire UdeM/GSASII/Ce3PtIn11.cif"
supercell = [120,120,30]

s=setStructure(all_q,i_q,mm,cif_location,supercell)




filename = "C:/Users/User/OneDrive - Universite de Montreal/Masters/Codes/MuESR_Outputs/XYZF_Ce3PtIn11_n20_q1bq2b.csv"
cells=[1,1,6]
#X,Y,Z,F,duration = getCubeFields(n=20,cells=cells,commensurate_type="s",nangles=0,timestamp=True)
#saveFile(filename,X,Y,Z,F,duration)
X,Y,Z,F,header=openSingleFile(filename)


# 109 G at (0,0.5,0.25)
# 109 G at (0.5,0,0.25)
# 0   G at (0.5,0.5,0.5)

# get index in X, Y and Z of the site of interest





# get value of field at this index in F





X,Y,Z,F = chooseInterval(X,Y,Z,F,116,135)
Xden,Yden,Zden,Fden=plot3d(X,Y,Z,F,a,b,c,cells=cells,normalized=True,atom_positions=atoms,filenameDFT = \
       "C:/Users/User/OneDrive - Universite de Montreal/Masters/Codes/MuESR_Outputs/density_Ce3PtIn11.txt"\
       ,dimDFT = [73,73,251],\
              title=r"Ce$_3$PtIn$_{11}$ Unit Cell"\
                     "\n"\
                     r"$q_1$ = ($\frac{1}{2},\frac{1}{2},\frac{1}{6}$), $q_2$ = ($\frac{1}{2},\frac{1}{2},0$), $m_c$ = 1 $\mu_B$,"\
                     " 116 $\pm$5 G"\
                     "\n")
