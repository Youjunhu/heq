#!/usr/bin/env python
# coding: utf-8
import MDSplus as MDS
import numpy as np
import matplotlib.pyplot as plt

EAST_server = MDS.Connection('mds.ipp.ac.cn') #'202.127.204.12'

def download_data(shot, treeName, signalName, with_dim=True):
    EAST_server.openTree(treeName, shot)
    signals = EAST_server.get(f"\{signalName}")
    if with_dim:
        time = EAST_server.get(f'dim_of(\{signalName})')
        print(f"{signalName}", EAST_server.get(f'units_of(\{signalName})'))
        return np.array(time, dtype='float32'), np.array(signals,dtype='float32')
    else:    
        return np.array(signals, dtype='float32')
nFL=35
nMP=38
nPFcoil=14
nFastCoil=2
nIp=1
n= nFL + nMP + nIp + nPFcoil + nFastCoil
#coil_turns = [120, 120, 120, 120, 120, 120, 248/5, 248/5, 248*4/5, 248*4/5, 60, 60, 32, 32]
coil_turns = [20*7, 20*7, 20*7, 20*7, 20*7, 20*7, 11*4, 11*4, 17*12, 17*12, 6*10, 6*10, 4*8, 4*8]
fast_coil_turns = 2

def magnetic_measurements(shot):
    data = [0]*n
    t = [0]*n
    k = 0
    for i in range(35): #flux loop measuements in PCS_EAST
        t[k], data[k] = download_data(shot=shot, treeName='PCS_EAST', signalName=f'PCFL{i+1}')
        #data[k]=data[k]/(2.*np.pi) #to Aphi*R
        k = k + 1
    for i in range(30): #equilibrium magnetic probe measurements
        t[k], data[k] = download_data(shot=shot, treeName='PCS_EAST', signalName=f'PCBPV{i+1}T')
        k = k + 1

    for i in [1,2,3,4,5,6,7,8]: #equilibrium magnetic probe measurements
        t[k], data[k] = download_data(shot=shot, treeName='PCS_EAST', signalName=f'PCBPL{i}T')
        k = k +1
    
    for i in range(nIp): # Plasma currents
        t[k], data[k] = download_data(shot=shot, treeName='EAST', signalName=f'IPE')
        data[k]=data[k]*1000 #from unit kA to Apere
        k = k + 1

    for i in range(nPFcoil): # PF coil currents
        #t[k], data[k] = download_data(shot=shot, treeName='EAST', signalName=f'PF{i+1}P')
        t[k], data[k] = download_data(shot=shot, treeName='pcs_EAST', signalName=f'PCPF{i+1}')
        data[k] = data[k]*coil_turns[i]
        k = k + 1
    for i in range(nFastCoil): # in-vessel coil controlling VDE (the fast coil)
        t[k], data[k] = download_data(shot=shot, treeName='pcs_EAST', signalName=f'pcic{i+1}')
        data[k] = data[k]*fast_coil_turns
        k = k + 1   
      
    return t, data


def get_poloidal_magnetic_flux(shot_number):
    time1, PSIRZ = download_data(shot=shot_number, treeName='efitrt_EAST', signalName=f'PSIRZ')
    time4, atime = download_data(shot=shot_number, treeName='efitrt_EAST', signalName=f'atime')
    return time4, PSIRZ

def one_discharge(shot):
    print(f'shot={shot}')
    time1, PSIRZ = get_poloidal_magnetic_flux(shot)
    print('PSIRZ.shape=',PSIRZ.shape)
    results = PSIRZ
    time2, mm = magnetic_measurements(shot)

    features = np.zeros((len(time1),len(mm)))
    
    print(f'input_shape={features.shape}, output_shape={results.shape}')

    for i, t0 in enumerate(time1):
       for j, d in enumerate(mm): #for each magnetic measurement
        tarray = time2[j]
        index = np.argmin(abs(tarray-t0)) #nearest neighbour interpolate
        features[i,j] = d[index]
    return features, results    

#time, PSIRZ = download_data(shot=shot, treeName='efitrt_EAST', signalName=f'PSIRZ')
#print(time)

shot=135101
f, r = one_discharge(shot)
print(f.shape, r.shape)
time_slice = f.shape[0]//2
np.savetxt('data/mm.txt',np.c_[f[time_slice]])
np.savetxt('data/psi.txt',r[time_slice])

R = download_data(shot=shot, treeName='efitrt_EAST', signalName=f'R', with_dim=False)
Z = download_data(shot=shot, treeName='efitrt_EAST', signalName=f'Z', with_dim=False)
print(R.shape)
print(Z.shape)
np.savetxt('data/r.txt', np.c_[R])
np.savetxt('data/z.txt', np.c_[Z])

print('plasma current(kA)=',f[time_slice,-17]/1000)
fl_no=np.array([i for i in range(35)]).astype(int)
np.savetxt('data/measured_flux.txt',np.c_[fl_no,f[time_slice][0:35]])
# time1, lcfs = download_data(shot=shot, treeName='efitrt_EAST', signalName=f'LIM')
# print(lcfs.shape)
# plt.plot(lcfs[:,0], lcfs[:,1], 'k.');
# np.savetxt('limiter.txt',np.c_[lcfs[:,0],lcfs[:,1]])
# plt.show()
