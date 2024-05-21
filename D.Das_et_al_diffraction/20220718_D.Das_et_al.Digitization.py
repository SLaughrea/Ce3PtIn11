# D. Das et al. Polycrystalline Sample Ce3PtIn11 X-Ray Diffraction Digitization
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.default'] = 'regular'


file_name = "wpd_datasets.csv"
data = np.genfromtxt(file_name,delimiter=",",skip_header =2)

I_calc_x,I_calc_y=data[:,0],data[:,1]
I_calc_x = I_calc_x[np.logical_not(np.isnan(I_calc_x))]
I_calc_y = I_calc_y[np.logical_not(np.isnan(I_calc_y))]
sorted(zip(I_calc_x,I_calc_y))
zip(*sorted(zip(I_calc_x,I_calc_y)))
I_calc_x,I_calc_y=zip(*sorted(zip(I_calc_x,I_calc_y)))
                      
I_obs_x,I_obs_y=data[:,2],data[:,3]
I_obs_x = I_obs_x[np.logical_not(np.isnan(I_obs_x))]
I_obs_y = I_obs_y[np.logical_not(np.isnan(I_obs_y))]
sorted(zip(I_obs_x,I_obs_y))
zip(*sorted(zip(I_obs_x,I_obs_y)))
I_obs_x,I_obs_y=zip(*sorted(zip(I_obs_x,I_obs_y)))
digitized=I_obs_x,I_obs_y
digitized = np.transpose(digitized)
np.savetxt("D.Das et al. digitization.xy",digitized)

I_obs_minus_I_calc_x,I_obs_minus_I_calc_y=data[:,4],data[:,5]
I_obs_minus_I_calc_x = I_obs_minus_I_calc_x[np.logical_not(np.isnan(I_obs_minus_I_calc_x))]
I_obs_minus_I_calc_y = I_obs_minus_I_calc_y[np.logical_not(np.isnan(I_obs_minus_I_calc_y))]
sorted(zip(I_obs_minus_I_calc_x,I_obs_minus_I_calc_y))
zip(*sorted(zip(I_obs_minus_I_calc_x,I_obs_minus_I_calc_y)))
I_obs_minus_I_calc_x,I_obs_minus_I_calc_y=zip(*sorted(zip(I_obs_minus_I_calc_x,I_obs_minus_I_calc_y)))

Ce3PtIn11_x,Ce3PtIn11_y=data[:,6],data[:,7]
Ce3PtIn11_x = Ce3PtIn11_x[np.logical_not(np.isnan(Ce3PtIn11_x))]
Ce3PtIn11_y = Ce3PtIn11_y[np.logical_not(np.isnan(Ce3PtIn11_y))]
Ce3PtIn11_y.fill(Ce3PtIn11_y[0])
sorted(zip(Ce3PtIn11_x,Ce3PtIn11_y))
zip(*sorted(zip(Ce3PtIn11_x,Ce3PtIn11_y)))
Ce3PtIn11_x,Ce3PtIn11_y=zip(*sorted(zip(Ce3PtIn11_x,Ce3PtIn11_y)))

In_x,In_y=data[:,8],data[:,9]
In_x = In_x[np.logical_not(np.isnan(In_x))]
In_y = In_y[np.logical_not(np.isnan(In_y))]
In_y.fill(In_y[0])
sorted(zip(In_x,In_y))
zip(*sorted(zip(In_x,In_y)))
In_x,In_y=zip(*sorted(zip(In_x,In_y)))

plt.figure(figsize = (10,8))
plt.text(10,1500,r"$Ce_3PtIn_{11}$",fontsize = 20)
plt.scatter(I_obs_x,I_obs_y,s=10,label=r"$I_{obs}$",color="red")
plt.plot(I_calc_x,I_calc_y,label=r"$I_{calc}$",color="black")
plt.plot(I_obs_minus_I_calc_x,I_obs_minus_I_calc_y,label=r"$I_{obs} - I_{calc}$",color="blue")
plt.scatter(Ce3PtIn11_x,Ce3PtIn11_y,label=r"$Ce_3PtIn_{11}$",color="green",marker="|")
plt.scatter(In_x,In_y,label=r"$In$",color="magenta",marker="|")
plt.xlabel(r"2$\theta (\degree$)",fontsize = 16)
plt.ylabel(r"$Intensity (counts)$",fontsize = 16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(5.2,80)
plt.ylim(-900,2000)
plt.legend(loc="upper right",fontsize = 20)
plt.savefig("D.Das et al. digitization.pdf")


