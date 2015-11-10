# RCM Sept 2015
# by special request from UZ
# and also as a programming exercise for RCM :)

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

p2 = 2.**np.arange(0,20)  # some powers of 2
#aufname = '/Users/ruth/Music/lossless/Chopin/1999 - Piano Concertos Nos. 1 & 2/01 01 - no.1 - 1 _ allegro maestoso.flac'
aufname = '/Users/ruth/Music/lossless/Pink Floyd/1975 - Wish You Were Here/02. Welcome To The Machine.flac'
data, samplerate = sf.read(aufname)[:]

# get some info about the data
N,tmp = np.shape(data)
dt = 1./samplerate
ti = np.arange(0,N*dt,dt)

# work out 1. the no of samples to include in each spectrum (Nwf)
#          2. the no of non-overlapping windows with Nwf samples in the series (Mw)
#          3. the no of windows to average over so we get a spectrum roughly every s (Mwa)

dtw = 1./20.            # range of human hearing is 20 Hz - 20 kHz
Nwf = np.round(dtw/dt)  # first guess window length for FFT
Nwfi = np.argmin(np.abs(Nwf - p2))
Nwf = p2[Nwfi]          # chosen window length (nearest power of 2)
dtw = Nwf*dt            # 1/lowest frequency resolvable in window
Mw = np.floor(N/Nwf)    # no of windows in series
Mwa = np.floor(1./dtw)  # no of windows to average over

# prepare data and perform FFTs
ts = np.mean(data,1)[0:Mw*Nwf].reshape(Mw,Nwf) # average over channels (?) then reshape array
han = np.hanning(Nwf)                          # Hanning window 
han = np.tile(han,(Mw,1))
ft = np.fft.fft(han*ts,axis=1)                 # FFT
sp = 2.*np.abs(ft)**2./(Nwf/samplerate)        # PSD
spg = np.reshape(sp[0:np.floor(Mw/Mwa)*Mwa,:],(np.floor(Mw/Mwa),Mwa,Nwf))
#np.shape(sp)
#Out[313]: (9590, 4096)
#np.shape(spg)
#Out[312]: (456, 21, 4096)
spa = np.mean(spg,axis=1)
var = np.sum(spa,axis=1)                        # dimensionless variance 
mvar = np.max(var)
tia = np.arange(0,np.floor(Mw/Mwa)*Nwf*dt*Mwa,Nwf*dt*Mwa)
f = np.fft.fftfreq(np.int(Nwf),dt)

# make a figure
fs = 10
fn = 'Times New Roman'
fig0 = plt.figure(1,figsize=(7,5),dpi=150,facecolor='white',edgecolor='black')
plt.show()

#for ii in range(0,len(tia)):
for ii in range(10,11):                

    t0 = tia[ii]    # [s]
    t0ai = np.argmin(np.abs(tia - t0))
    t0i = np.argmin(np.abs(ti - t0))

    ax00 = fig0.add_axes([0.1, 0.1, 0.8, 0.55])
    pc00 = ax00.pcolorfast(tia/60,f[f>0]/1000,np.transpose(10.*np.log10(spa[:,f>0]/mvar)))
    pc00.set_clim(-20,-120)
    pc00.set_cmap('nipy_spectral')
    pl01 = ax00.plot(np.array([t0,t0])/60,np.array([0,np.max(f)/1000]),'k--',linewidth=0.7)
    pl02 = ax00.plot(np.array([t0,t0])/60-0.5,np.array([0,np.max(f)/1000]),'k--',linewidth=0.4)
    pl03 = ax00.plot(np.array([t0,t0])/60+0.5,np.array([0,np.max(f)/1000]),'k--',linewidth=0.4)
    
    #ax00.set_title('Time = %.01f hr' % tis,fontsize=fs,fontname=fn)
    ax00.set_xticks(np.arange(tia[0],tia[-1]/60,1.))
    ax00.set_xticklabels(np.arange(tia[0],tia[-1]/60,1.),fontsize=fs,fontname=fn)
    ax00.set_xlim(0,tia[-1]/60)
    ax00.set_xlabel('Time (min)',fontsize=fs,fontname=fn)
    #ax00.xax00is.set_label_coords(0.5,-0.08)
    ax00.set_yticks(np.arange(0,np.max(f)/1000,2))
    ax00.set_yticklabels(np.arange(0,np.max(f)/1000,2),fontsize=fs,fontname=fn)    
    ax00.set_ylabel('Frequency (kHz)',fontsize=fs,fontname=fn)
    
    cax = fig0.add_axes([0.92, 0.1, 0.02, 0.8])
    cb = fig0.colorbar(pc00,cax)
    for tmp in cax.get_yticklabels():
        tmp.set_fontname(fn)
        tmp.set_fontsize(fs)

    ax01 = fig0.add_axes([0.1, 0.7, 0.8, 0.2])
    pl00 = ax01.plot(ti[1:-1:10]/60.,np.mean(data,1)[1:-1:10])
    ax01.set_xticks(np.arange(t0/60-0.5,t0/60+0.5,0.1))
    ax01.set_xticklabels(np.arange(t0/60-0.5,t0/60+0.5,0.1),fontsize=fs,fontname=fn)
    ax01.set_xlim(t0/60.-0.5,t0/60.+0.5)
    ax01.set_ylim(-0.4,0.4)
    ax01.set_yticks(np.arange(-0.4,0.5,0.1))
    ax01.set_yticklabels(np.arange(-0.4,0.5,0.1),fontsize=fs,fontname=fn)
    ax01.grid(True)
    majorFormatter = plt.FormatStrFormatter('%.1f')
    majorxFormatter = plt.FormatStrFormatter('%.2f')
    ax01.yaxis.set_major_formatter(majorFormatter)
    ax01.xaxis.set_major_formatter(majorxFormatter)
    te01 = ax01.text(-0.4,0.65,aufname,fontsize=fs-2,fontname=fn)
    te02 = ax01.text(-0.4,0.5,'samplerate = %g s$^{-1}$' % samplerate,fontname=fn,fontsize=fs-2)
    
        # pl000 = ax00.plot(f,np.mean(ft,axis=1))
    # ax00.set_yscale('log')
    # ax00.set_xscale('log')
    
    plt.draw()
    fname = 'mov/mov%02d.png' % ii
    plt.savefig(fname,dpi=300)

#    fig0.delaxes(ax00)
#    fig0.delaxes(ax01)
#    fig0.delaxes(cax)
#plt.plot(ti[0:-1:1000],data[0:-1:1000,0],'k')
