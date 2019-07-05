import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np


C=3*(10**8)
Fmax=2.47*(10**9)
Fmin=2.34*(10**9)
BW=Fmax-Fmin
lamda=(2*C)/(Fmax+Fmin)
# Read wav file
def read_wav(kalyan):
    [Fs,data] = wavfile.read(kalyan)
    print('Sampling frequency Fs is :', Fs)
    return [Fs,data]
[Fs,data] = read_wav('Rec_2ch_5_away_back_fast.wav')

print("The length of the the data  is :",len(data))
#print(Y)
limiteddata=data[:2000]# To reduce the length to first 2000
print("The length of the the new data is :",len(limiteddata))
#print(Y)


#sync pulse and reflected signal data extraction limited t0 2000 sa0ples
sync_pulse=limiteddata[:,0]   
reflected_signal=limiteddata[:,1]
#sync_pulse and reflected siganl data extraction FULL
sp_full=data[:,0]
reflected_full=data[:,1]

#eliminating the first 100 values in sync_pulse and reflected signal
sp_full=sp_full[99:]
reflected_full=reflected_full[99:]
#FIGURE 

plt.figure(1)
t=np.arange(0,len(limiteddata))
plt.plot(t,sync_pulse,'b',t,reflected_signal,'r')
plt.title("Data plot")
plt.xlabel("xaxis")
plt.ylabel("amplitude")
plt.grid(True)
plt.show()

#the following code is to separate the even sweeps and odd sweeps
sweeps=[]

temp_Sweeps=[]
count=0

if sp_full[0] < 0:
    Ini_sign_Flag = -1
elif sp_full[0] > 0:
    Ini_sign_Flag = 1
else:
    Ini_sign_Flag = 0

temp_Sweeps.append(reflected_full[0])

for i in range(1, len(sp_full)):
    if sp_full[i] < 0:
        sign_Flag = -1
    elif sp_full[i] > 0:
        sign_Flag = 1
    else:
        sign_Flag = 0
        
#    if sign_Flag != 0:
    if Ini_sign_Flag==sign_Flag:
        temp_Sweeps.append(reflected_full[i])
    else:
        sweeps.append(temp_Sweeps)
        count=+count
        temp_Sweeps=[]
        temp_Sweeps.append(reflected_full[i])
    
    Ini_sign_Flag=sign_Flag

sweeps.append(temp_Sweeps)

window=[]
for i in range(len(sweeps)):
    window.append(np.hanning(len(sweeps[i])))
    
windowed_data=[]    
for i in range(len(sweeps)):
    windowed_data.append(window[i]*sweeps[i])
    
paded_data=[]
for i in range(len(windowed_data)):
    paded_data.append(np.pad(windowed_data[i], (0, 1024 - len(windowed_data[i])%1024), 'constant'))
    #paded_data.append(np.pad(windowed_data[i], (1024 - len(windowed_data[i])%1024,1023), 'constant'))

fft_data=[]
for i in range(len(paded_data)):
    #sample_padeddata= paded_data[i]
    #fft_data.append(np.fft.fft(sample_padeddata))
    fft_data.append(np.fft.fft( paded_data[i]))
 
halfof_fftdata=[]
sample_fftdata=[]
for i in range(len(fft_data)):
    sample_fftdata= fft_data[i]
    halfof_fftdata.append(sample_fftdata[0:511])    
    
#Spherical plotting 
print('SPHERICAL SPREADING')
alpha= 2
sample_data = []
spherical_output=[]
for i in range(len(halfof_fftdata)):
    sample_data = halfof_fftdata[i]
    for j in range(len(sample_data)):
        sample_data[j]=sample_data[j] * pow(j,alpha)
        
    halfof_fftdata[i]=sample_data
    plt.plot(halfof_fftdata[i])    
 
#doing the absolute          
abs_data=[]
for i in range(len(halfof_fftdata)):
    fft_data=halfof_fftdata[i]
    fft_data=fft_data[0:39]
    the_data=[]
    for j in range(len(fft_data)):
        the_data.append(np.real(np.abs(fft_data[j])))
        
    halfof_fftdata[i] = the_data    


#heat plot             
def imagesc(data,maxtime,maxrange):
    fig, ax = plt.subplots()
    amax = int(np.max(data))
    amin = int(np.min(data))
    print('amp min ', amin, 'amp max ', amax)
    im = ax.imshow(data, cmap=plt.get_cmap('hot'),
    interpolation='nearest', vmin=amin, vmax=amax,
    aspect='auto',extent=[0,maxrange,maxtime,0])
    fig.colorbar(im)
    plt.xlabel('Range[m]')
    plt.ylabel('Time[s]')
    plt.show()
    return


imagesc(halfof_fftdata,40,40)


#for fig 4
#sum of the data
sumof_rows=np.sum(halfof_fftdata,axis=0)#sum of rows 

distance=[]

for i in range(0,39):
     distance.append((C/(4*BW))*(i))

plt.figure(2)  
plt.title("Distance plot ")  
plt.plot(distance,sumof_rows)
plt.xlabel("distance in meters")
plt.ylabel("amplitude")    


#for fig 5
#diff of the data
evenrows=np.zeros(39)
oddrows=np.zeros(39)
for i in range(1,len(halfof_fftdata)):
    evenrows=evenrows+halfof_fftdata[i]
    i=i+1

for i in range(0,len(halfof_fftdata)):
    oddrows=oddrows+halfof_fftdata[i]
    i=i+1   

'''
evenrows=halfof_fftdata[1::2]#separating the even rows
evenrows=np.sum(evenrows,axis=0)# adding all rows 
oddrows=halfof_fftdata[::2]#separating the odd rows
oddrows=np.sum(oddrows,axis=0)#adding all rows
'''
difference=np.zeros(39)
difference=oddrows-evenrows# has the difference data

velocity=[]
for i in range(0,39):
    velocity.append((lamda/4)*(i))

plt.figure(3)
plt.title("velocity plot")
plt.plot(velocity,difference)
plt.xlabel("velocity in meters per sec")
plt.ylabel("amplitude")
