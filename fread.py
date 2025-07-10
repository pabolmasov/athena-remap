from numpy import *
from matplotlib import gridspec

import os
import sys
import glob
sys.path.append("vis/python")
# sys.path.append("/Users/pasha/athena/vis/python")
import athena_read

import h5py

# visualization routines for field_loop_testtint

# from scipy.optimize import root_scalar
# from scipy.integrate import simpson
# from scipy.integrate import cumulative_trapezoid as cumtrapz

from os.path import exists

ifplot = True
if(ifplot):
    import matplotlib
    from matplotlib.pyplot import *
    
cmap = 'jet'

# reading hydro variables:
def qplot(nfile, ddir = 'models/loop', qua = 'rho', prefix = 'Loop.prim', zslice = 0., domainnet = True):

    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = [qua], raw = True)

    time = data['Time']

    levels = data['Levels']
    print("maximal level is ", levels.max())
    nlevels=size(levels)
    print(nlevels, " blocks in total")
    
    q = data[qua]
    
    vmin = quantile(q, 0.01)
    vmax = quantile(q, 0.99)
    
    clf()
    
    for k in arange(nlevels):
        x = data['x1v'][k,:]
        y = data['x2v'][k,:]
        z = data['x3v'][k,:]
        
        dz = z[1]-z[0]

        #print("z = ", z.min(), "..", z.max())
        if (z.max() >= zslice) and (z.min() <= zslice):
        
            kzslice = (z[z<=zslice]-zslice).argmax()
            q1 = data[qua][k,kzslice, :, :] ;  q2 = data[qua][k,kzslice+1, :, :]
            q = (q2-q1) * (zslice - z[kzslice])/dz + q1
            
            pc = pcolormesh(x, y, q, vmin=vmin, vmax=vmax, cmap = cmap)
            print("q = ", q.min(), "..", q.max())
            if domainnet:
                plot([x.min(), x.max()], [y.min(), y.min()], 'w:')
                plot([x.min(), x.max()], [y.max(), y.max()], 'w:')
                plot([x.min(), x.min()], [y.min(), y.max()], 'w:')
                plot([x.max(), x.max()], [y.min(), y.max()], 'w:')

    xlabel('$X$')  ;  ylabel('$Y$')
    colorbar(pc)
    title(r'$t = '+str(round(time, 2))+'$')
    savefig(ddir+'/'+qua+'{:05d}'.format(nfile)+'.png') #, DPI=500)

# B components:
def Bplot(nfile, ddir = 'models/loop', prefix = 'Loop.prim', zslice = 0.1, domainnet = True):

    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['Bcc1', 'Bcc2', 'Bcc3'], raw = True)

    time = data['Time']

    levels = data['Levels']
    print("maximal level is ", levels.max())
    nlevels=size(levels)
    print(nlevels, " blocks in total")

    fig, (ax0, ax1, ax2, ax3) = subplots(nrows=1, ncols=4, width_ratios = (1,1,1,0.1),
                                    figsize=(15, 4))
                                    
    Bx = data['Bcc1'] ; By = data['Bcc2'] ; Bz = data['Bcc3']
    
    vmin = quantile(minimum(minimum(Bx,By),Bz), 0.01)
    vmax = quantile(maximum(maximum(Bx,By),Bz), 0.99)

    #     vmin = -1e-7 ; vmax = 1e-7
    
    # number of NaN points:
    print("n(Bx==NaN) = ", isnan(Bx).sum())
    print("n(By==NaN) = ", isnan(By).sum())    
    print("n(Bz==NaN) = ", isnan(Bz).sum())
    
    for k in arange(nlevels):
        x = data['x1v'][k,:]
        y = data['x2v'][k,:]
        z = data['x3v'][k,:]
        
        dz = z[1]-z[0]

        #print("z = ", z.min(), "..", z.max())
        if (z.max() > zslice) and (z.min() < zslice):
        
            kzslice = (z[z<=zslice]-zslice).argmax()
            # print("kz = ", kzslice)
            
            Bx1 = data['Bcc1'][k,kzslice, :, :] ;  Bx2 = data['Bcc1'][k,kzslice+1, :, :]
            By1 = data['Bcc2'][k,kzslice, :, :] ;  By2 = data['Bcc2'][k,kzslice+1, :, :]
            Bz1 = data['Bcc3'][k,kzslice, :, :] ;  Bz2 = data['Bcc3'][k,kzslice+1, :, :]
        
            Bx = (Bx2-Bx1) * (zslice - z[kzslice])/dz + Bx1
            By = (By2-By1) * (zslice - z[kzslice])/dz + By1
            Bz = (Bz2-Bz1) * (zslice - z[kzslice])/dz + Bz1
            
            pcx = ax0.pcolormesh(x, y, Bx, vmin=vmin, vmax=vmax, cmap = cmap)
            pcy = ax1.pcolormesh(x, y, By, vmin=vmin, vmax=vmax, cmap = cmap)
            pcz = ax2.pcolormesh(x, y, Bz, vmin=vmin, vmax=vmax, cmap = cmap)
            
            if domainnet:
                ax0.plot([x.min(), x.max()], [y.min(), y.min()], 'w:')
                ax0.plot([x.min(), x.max()], [y.max(), y.max()], 'w:')
                ax0.plot([x.min(), x.min()], [y.min(), y.max()], 'w:')
                ax0.plot([x.max(), x.max()], [y.min(), y.max()], 'w:')
                ax1.plot([x.min(), x.max()], [y.min(), y.min()], 'w:')
                ax1.plot([x.min(), x.max()], [y.max(), y.max()], 'w:')
                ax1.plot([x.min(), x.min()], [y.min(), y.max()], 'w:')
                ax1.plot([x.max(), x.max()], [y.min(), y.max()], 'w:')
                ax2.plot([x.min(), x.max()], [y.min(), y.min()], 'w:')
                ax2.plot([x.min(), x.max()], [y.max(), y.max()], 'w:')
                ax2.plot([x.min(), x.min()], [y.min(), y.max()], 'w:')
                ax2.plot([x.max(), x.max()], [y.min(), y.max()], 'w:')
            # print("Bz = ", Bz.min(), "..", Bz.max())
    ax0.set_title('$B_x$') ; ax1.set_title('$B_y$') ; ax2.set_title('$B_z$')
    ax0.set_xlabel('$X$') ; ax1.set_xlabel('$X$') ; ax2.set_xlabel('$X$')
    ax0.set_ylabel('$Y$')
    colorbar(pcx, cax = ax3)
    fig.suptitle(r'$t = '+str(round(time, 2))+'$')
    savefig(ddir+'/B'+'{:05d}'.format(nfile)+'.png') #, DPI=500)


def divCooked(nfile, ddir = 'loop', prefix = 'Loop.b'):
    
    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['B1', 'B2', 'B3'], raw = False, num_ghost=0)

    x = data['x1v']
    y = data['x2v']
    z = data['x3v']

    dx = x[1]-x[0] ; dy = y[1]-y[0] ; dz = z[1]-z[0]
    
    Bx = data['B1'][:, :, :]  ; By = data['B2'][:, :, :]  ; Bz = data['B3'][:, :, :]

    dB = (Bx[:-1, :-1, 1:]-Bx[:-1, :-1, :-1])/dx + (By[:-1, 1:, :-1]-By[:-1, :-1, :-1])/dy + (Bz[1:, :-1, :-1]-Bz[:-1, :-1, :-1])/dz

    dBmax = abs(dB).max(axis = 0)

    bsqmax = (Bx**2+By**2+Bz**2).max()
    
    clf()
    pc = pcolormesh(x, y, log10(abs(dB/sqrt(bsqmax*dx))), vmin=vmin, vmax=vmax, cmap = cmap)
    cb = colorbar(pc)
    cb.set_label(r'$\nabla \cdot B$')
    xlabel(r'$X$')  ; ylabel(r'$Y$')
    title(r'$t = '+str(round(time, 2))+'$')
    savefig(ddir+'/Cdiv'+'{:05d}'.format(nfile)+'.png') #, DPI=500)

def divX(nfile, ddir ='loop', prefix = 'Loop.b'):
    
    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['B1', 'B2', 'B3'], raw = True)

    time = data['Time']

    levels = data['Levels']
    print("maximal level is ", levels.max())
    nlevels=size(levels)
    print(nlevels, " blocks in total")

    x = data['x1f']
    nx = shape(x)[1]
    
    divYZ = zeros([nlevels, nx-2])
    
    clf()
    fig, ax = subplots(2,1)

    formatsequence1 = ['ko', 'ro', 'go', 'bo'] ; formatsequence2 = ['k*', 'r*', 'g*', 'b*']

    divmax = 0.
    
    for k in arange(nlevels):
        x = data['x1f'][k,:] ; y = data['x2f'][k,:] ; z = data['x3f'][k,:]
        dx = x[1]-x[0]  ; dy = y[1] - y[0] ; dz = z[1] - z[0]
        Bx = data['B1'][k,:,:,:] ; By = data['B2'][k,:,:,:] ; Bz = data['B3'][k,:,:,:] 
        divB = (Bx[:-1, :-1, 1:]-Bx[:-1, :-1,:-1])/dx + (By[:-1, 1:, :-1]-By[:-1, :-1,:-1])/dy + (Bz[1:, :-1, :-1]-Bz[:-1, :-1,:-1])/dz
    
        ax[0].plot(x[1:-1], Bx.std(axis = 0).std(axis = 0)[1:], formatsequence1[levels[k]])
        ax[0].plot(x[1:-1], By.std(axis = 0).std(axis = 0)[:-1], formatsequence2[levels[k]])

        ax[1].plot(x[1:-1], abs(divB).max(axis = 0).max(axis = 0), formatsequence1[levels[k]])
        divmax = maximum(divmax,abs(divB).max())

    ax[0].set_xlabel(r'$X$') ; ax[1].set_xlabel(r'$X$') 
    ax[0].set_yscale('log') ; ax[1].set_yscale('log') ;ax[0].set_ylabel(r'$B$') ; ax[1].set_ylabel(r'$\nabla \cdot B$')
    ax[0].set_ylim(1e-15,1e-5) ; ax[1].set_ylim(1e-15,divmax)
    fig.set_size_inches(8.,8.)
    savefig('divX.png')
    
def divY(nfile, ddir ='loop', prefix = 'Loop.b'):
    
    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['B1', 'B2', 'B3'], raw = True)

    time = data['Time']

    levels = data['Levels']
    print("maximal level is ", levels.max())
    nlevels=size(levels)
    print(nlevels, " blocks in total")

    y = data['x2f']
    ny = shape(y)[1]
    
    #    divXY = zeros([nlevels, nx-2])

    divmax = 0.
    
    clf()
    fig, ax = subplots(2,1)

    formatsequence1 = ['ko', 'ro', 'go', 'bo'] ; formatsequence2 = ['k*', 'r*', 'g*', 'b*']
    
    
    for k in arange(nlevels):
        x = data['x1f'][k,:] ; y = data['x2f'][k,:] ; z = data['x3f'][k,:]
        dx = x[1]-x[0]  ; dy = y[1] - y[0] ; dz = z[1] - z[0]
        Bx = data['B1'][k,:,:,:] ; By = data['B2'][k,:,:,:] ; Bz = data['B3'][k,:,:,:] 
        divB = (Bx[:-1, :-1, 1:]-Bx[:-1, :-1,:-1])/dx + (By[:-1, 1:, :-1]-By[:-1, :-1,:-1])/dy + (Bz[1:, :-1, :-1]-Bz[:-1, :-1,:-1])/dz
    
        ax[0].plot(y[1:-1], Bx.std(axis = 0).std(axis = -1)[1:], formatsequence1[levels[k]])
        ax[0].plot(y[1:-1], By.std(axis = 0).std(axis = -1)[:-1], formatsequence2[levels[k]])

        ax[1].plot(y[1:-1], abs(divB).max(axis = 0).max(axis = -1), formatsequence1[levels[k]])
        divmax = maximum(divmax,abs(divB).max())

    ax[0].set_xlabel(r'$Y$') ; ax[1].set_xlabel(r'$Y$') 
    ax[0].set_yscale('log') ; ax[1].set_yscale('log') ;ax[0].set_ylabel(r'$B$') ; ax[1].set_ylabel(r'$\nabla \cdot B$')
    ax[0].set_ylim(1e-15,1e-2) ; ax[1].set_ylim(1e-15,divmax)
    fig.set_size_inches(8.,8.)
    savefig('divY.png')
    
   
def divZ(nfile, ddir ='loop', prefix = 'Loop.b'):

    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['B1', 'B2', 'B3'], raw = True)

    time = data['Time']

    levels = data['Levels']
    print("maximal level is ", levels.max())
    nlevels=size(levels)
    print(nlevels, " blocks in total")

    z = data['x3f']
    nz = shape(z)[1]
    
    divXY = zeros([nlevels, nz-2])

    divmax = 0.
    
    clf()
    fig, ax = subplots(2,1)

    formatsequence1 = ['ko', 'ro', 'go', 'bo'] ; formatsequence2 = ['k*', 'r*', 'g*', 'b*']
    
    for k in arange(nlevels):
        x = data['x1f'][k,:] ; y = data['x2f'][k,:] ; z = data['x3f'][k,:]
        dx = x[1]-x[0]  ; dy = y[1] - y[0] ; dz = z[1] - z[0]
        Bx = data['B1'][k,:,:,:] ; By = data['B2'][k,:,:,:] ; Bz = data['B3'][k,:,:,:] 
        divB = (Bx[:-1, :-1, 1:]-Bx[:-1, :-1,:-1])/dx + (By[:-1, 1:, :-1]-By[:-1, :-1,:-1])/dy + (Bz[1:, :-1, :-1]-Bz[:-1, :-1,:-1])/dz
    
        ax[0].plot(z[1:-1], Bx.std(axis = -1).std(axis = -1)[1:], formatsequence1[levels[k]])
        ax[0].plot(z[1:-1], By.std(axis = -1).std(axis = -1)[:-1], formatsequence2[levels[k]])

        ax[1].plot(z[1:-1], abs(divB).max(axis = -1).max(axis = -1), formatsequence1[levels[k]])
        divmax = maximum(divmax,abs(divB).max())

    ax[0].set_xlabel(r'$Z$') ; ax[1].set_xlabel(r'$Z$') 
    ax[0].set_yscale('log') ; ax[1].set_yscale('log') ;ax[0].set_ylabel(r'$B_z$') ; ax[1].set_ylabel(r'$\nabla \cdot B$')
    ax[0].set_ylim(1e-15,1e-5) ; ax[1].set_ylim(1e-15,divmax)
    fig.set_size_inches(8.,8.)
    savefig('divZ.png')
        
def divBplot(nfile, ddir = 'loop', prefix = 'Loop.b', zslice = None, domainnet = True, vmin = -20., vmax = -10., ghost = 2, xslice = None, minlevel = 0):
    
    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['B1', 'B2', 'B3'], raw = True)

    time = data['Time']

    levels = data['Levels']
    print("maximal level is ", levels.max())
    nlevels=size(levels)
    print(nlevels, " blocks in total")

    print(shape(data['B1']))
    print(shape(data['x1f']))
  
    absmax = 0.
    bsqmax = 0.
  
    clf()
  
    for k in arange(nlevels):
        x = data['x1v'][k,:]
        y = data['x2v'][k,:]
        z = data['x3v'][k,:]
        xf = data['x1f'][k,:]
        yf = data['x2f'][k,:]
        zf = data['x3f'][k,:]

        nx = size(x)
        # print(zslice)

        if (zslice is None) or ((zf.max() > zslice) and (zf.min() < zslice)):
            dz = z[1] - z[0] ; dy = y[1] - y[0] ; dx = x[1] - x[0]
            if zslice is None:
                Bx = data['B1'][k,:, :, :]  ; By = data['B2'][k,:, :, :]  ; Bz = data['B3'][k,:, :, :]

                bsq = Bx**2+By**2+Bz**2
                bsq = bsq[:-1, :-1, :-1]
                
                divB = (Bx[:-1,:-1,1:]-Bx[:-1,:-1,:-1])/dx + (By[:-1,1:,:-1]-By[:-1,:-1,:-1])/dy + (Bz[1:,:-1,:-1]-Bz[:-1,:-1,:-1])/dz
                if (ghost > 0):
                    divB = divB[ghost+1:-ghost, ghost+1:-ghost, ghost+1:-ghost]
                    bsq = bsq[ghost+1:-1-ghost, ghost+1:-ghost, ghost+1:-ghost]
                wz = abs(divB.max(axis=-1).max(axis=-1)).argmax(axis=0)
                q = abs(divB)
                bsqmax = maximum(bsq.max()/dx**2, bsqmax) # B^2 / dx^2

                # print(shape(q))
                # ii = input('q')
                qx = q.max(axis = 0)
                # if qx.max() > -1.:
                # print("max X = ", x[qx.argmax()], ", lgdivB = ", qx.max())
            else:
            
                kzslice = (zf[zf<zslice]).argmax()
            
                Bx1 = data['B1'][k,kzslice, :, :] ;  Bx2 = data['B1'][k,kzslice+1, :, :]
                By1 = data['B2'][k,kzslice, :, :] ;  By2 = data['B2'][k,kzslice+1, :, :]
                Bz1 = data['B3'][k,kzslice, :, :] ;  Bz2 = data['B3'][k,kzslice+1, :, :]

                bsq = Bx1**2+By1**2+Bz1**2
                bsqmax = maximum(bsq.max()/dx**2, bsqmax) # B^2 / dx^2
                # print(shape(Bx1), shape(By1), shape(Bz1))

                divB = (Bx1[:-1,1:]-Bx1[:-1,:-1])/dx + (By1[1:,:-1]-By1[:-1,:-1])/dy + (Bz2[:-1,:-1]-Bz1[:-1,:-1])/dz
                if ghost > 0:
                    divB = divB[ghost:-ghost, ghost:-ghost]
                    bsq = bsq[ghost:-1-ghost, ghost:-1-ghost]
                else:
                    bsq = bsq[:-1, :-1]
                q = abs(divB)
                qx = abs(divB) # /sqrt(bsq.max())*dx)

            if (xslice is not None):

                if ((x<xslice).sum() > 0) & ((x>xslice).sum() > 0):
                
                    kxslice = (x[x<xslice]).argmax()
                    kxslice = minimum(kxslice, nx-1)
                    # if x.min() < -1.7 and x.min() > -1.9:
                    # kxslice = 0
                    if (levels[k]>= minlevel) & (kxslice < nx-1):
                        print(shape(yf), shape(zf), shape(divB))
                        # pc = pcolormesh(y[1:], z[1:], log10(abs(squeeze(divB[:,:,kxslice])/sqrt(bsq.max())*dx)), vmin=vmin, vmax=vmax, cmap = cmap)
                        pc = pcolormesh(y[1:], z[1:], log10(abs(squeeze(divB[:,:,kxslice]))), vmin=vmin, vmax=vmax, cmap = cmap)
                        # print(vmin, vmax)
                        print("x = ", x[kxslice], "max divB = ", log10(abs(divB[:,:,kxslice])).max())
                        # ii = input('X')

                    if domainnet and (xslice is not None):
                        plot([y.min(), y.max()], [z.min(), z.min()], 'g:')
                        plot([y.min(), y.max()], [z.max(), z.max()], 'g:')
                        plot([y.min(), y.min()], [z.min(), z.max()], 'g:')
                        plot([y.max(), y.max()], [z.min(), z.max()], 'g:')
                         
            else:
                # print(shape(qx), shape(bsq), shape(x[ghost:-1-ghost]))
                if (levels[k] >= minlevel) & (xslice is None):
                    if ghost > 0:
                        pc = pcolormesh(x[ghost:-ghost-1], y[ghost:-ghost-1], log10(qx/sqrt(bsq).max(axis = 0)), vmin=vmin, vmax=vmax, cmap = cmap)
                    else:
                        # print(shape(qx), shape(bsq), shape(x[:-1]))
                        pc = pcolormesh(x[:-1], y[:-1], log10(qx/sqrt(bsq).max(axis=0)), vmin=vmin, vmax=vmax, cmap = cmap)
                        # print("block ", k, ": max(abs(divB)) = ", abs(divB/sqrt(bsq.max())*dx).max())
            if (zslice is None) and (q.max() > 1e-10):
                print("z(max) = ", z[wz], " ( level ", levels[k], ")")
                
            # print(shape(q))
            absmax = maximum(absmax, (q).max())
            
            if domainnet and (xslice == None):
                plot([x.min(), x.max()], [y.min(), y.min()], 'g:')
                plot([x.min(), x.max()], [y.max(), y.max()], 'g:')
                plot([x.min(), x.min()], [y.min(), y.max()], 'g:')
                plot([x.max(), x.max()], [y.min(), y.max()], 'g:')
           

    if absmax > 0.0:
        cb = colorbar(pc)
        cb.set_label(r'$\nabla \cdot B$')
    if xslice is not None:
        xlabel(r'$Y$')  ; ylabel(r'$Z$')
        title(r'$t = '+str(round(time, 2))+'$, $X = '+str(xslice)+'$')
        savefig(ddir+'/divX'+'{:05d}'.format(nfile)+'.png') #, DPI=500)
    else:
        xlabel(r'$X$')  ; ylabel(r'$Y$')
        title(r'$t = '+str(round(time, 2))+'$')
        savefig(ddir+'/div'+'{:05d}'.format(nfile)+'.png') #, DPI=500)

    print("max |B| = ", sqrt(bsqmax))
        
    return time, (absmax/sqrt(bsqmax))


def divBmovie(narr, ddir = 'models/loop', prefix = 'Loop'):

    nnarr = size(narr)
    divBmax = zeros(nnarr)
    time = zeros(nnarr)

    for k in arange(nnarr):
        Bplot(narr[k], ddir=ddir, prefix=prefix+'.prim', zslice  = 0.05, domainnet = True)
        dtmp = divBplot(narr[k], ddir = ddir, prefix = prefix+'.b', ghost=0, vmin = -15, vmax = -2)
        divBmax[k] = dtmp[1]
        time[k] = dtmp[0]
        
    clf()
    plot(time, divBmax, 'k.')
    yscale('log')
    xlabel(r'$t$') ; ylabel(r'$\Delta x \frac{\max |\nabla \cdot \mathbf{B}|}{\sqrt{\max B^2}}$')
    title(ddir)
    savefig(ddir+'/curvedivBplot.png')

def plot_diff(B1, B2, y, z, fname = 'dBtest'):

    dx = y[1] - y[0]
    dB = abs(B1-B2)/(abs(B1).max()+abs(B2).max()) * dx
    # clf()
    fig1, ax1 = subplots(1,1)
    pc = ax1.pcolormesh(y, z, log10(dB), vmin = log10(dB[dB>1e-20].min()), vmax = log10(dB.max()))
    fig1.colorbar(pc)
    ax1.set_xlabel(r'$Y$')   ;        ax1.set_ylabel(r'$Z$')
    savefig(fname+'.png')
    close()
    print("max dBnorm = ", dB.max())

def intersections(nfile, ddir = 'models/loop', prefix = 'Loop', ghost = 2, Bxcutoff = 1e-6):

    filename = ddir+'/'+prefix+'.b.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['B1', 'B2', 'B3'], raw = True)
    
    levels = data['Levels']
    nlevels = size(levels)

    xmin = zeros(nlevels) ; xmax = zeros(nlevels)
    ymin = zeros(nlevels) ; ymax = zeros(nlevels)
    zmin = zeros(nlevels) ; zmax = zeros(nlevels)

    xref = -0.7 ; yref = -0.309375 ; zref = 0.309375
    
    # collecting the data about the boxes
    for k in arange(nlevels):
        xf = data['x1f'][k,:]    ;    yf = data['x2f'][k,:]  ;      zf = data['x3f'][k,:]
        xv = data['x1v'][k,:]    ;    yv = data['x2v'][k,:]  ;      zv = data['x3v'][k,:]
        nx = size(xf) ; ny = size(yv)  ; nz = size(zv) 
        dx = xf[1]-xf[0] ; dy = yf[1]-yf[0]  ; dz = zf[1]-zf[0]
        # print(shape(x))
        xmin[k] = xf[ghost] ;  xmax[k] = xf[-ghost-1]
        ymin[k] = yf[ghost] ;  ymax[k] = yf[-ghost-1]
        zmin[k] = zf[ghost] ;  zmax[k] = zf[-ghost-1]

        wx = (abs(xf - xref) < 1e-5)
        wy = (abs(yv - yref-dy/2.) < 1e-5)
        wz = (abs(zv - zref-dz/2.) < 1e-5)

        if ((wx).sum()*(wy).sum()*(wz).sum()) > 0:
            # print(zf)
            print(arange(nx)[wx], arange(ny)[wy], arange(nz)[wz])
            kx = (arange(nx)[wx])[0]  ; ky = (arange(ny)[wy])[0]  ; kz =(arange(nz)[wz])[0]
            if levels[k] == 0:
                Bx = data['B1'][k,kz,ky,kx]
            else:
                print(data['B1'][k,kz:kz+2,ky:ky+2,kx])
                Bx = (data['B1'][k,kz:kz+2,ky:ky+2,kx]).mean()
            print("k = ", k, ": B at refpoint = ", Bx)
        # xmin[k] = x.min()   ;     xmax[k] = x.max()
        # ymin[k] = y.min()   ;     ymax[k] = y.max()
        # zmin[k] = z.min()   ;     zmax[k] = z.max()

    # print(xmin, xmax)
    # xmin2, xmax2 = meshgrid(xmin, xmax)
    # print(abs(xmin2-xmax2).min())
    ii = input('X')
        
    rightlist = [] # the list of indices of the meshblocks located to the right (next in X direction)
    nk = zeros(nlevels, dtype = int)
    
    for k in arange(nlevels):
        print(abs(xmin-xmax[k]).min())
        w = (xmin == xmax[k]) & ((ymin+ymax) <= 2.*ymax[k]) & ((ymin+ymax) >= 2.*ymin[k]) & ((zmin+zmax) <= 2.*zmax[k]) & ((zmin+zmax) >= 2.*zmin[k]) 
        # (xmin <= xmax[k]) & (xmin > xmin[k]) & ((ymin+ymax) <= 2.*ymax[k]) & ((ymin+ymax) >= 2.*ymin[k]) & ((zmin+zmax) <= 2.*zmax[k]) & ((zmin+zmax) >= 2.*zmin[k]) 
        # & (ymin <= ymax[k]) & (ymax >= ymin[k]) & (zmin <= zmax[k]) & (zmax >= zmin[k]) & (arange(nlevels) != k)
        nk[k] = w.sum()
        rightlist.append(arange(nlevels)[w])
        print("k: ", arange(nlevels)[w])
    
    ii = input('W')
            
    for k in arange(nlevels):

        xleft = squeeze(data['x1f'][k,:])
        yleft = squeeze(data['x2v'][k,:])
        zleft = squeeze(data['x3v'][k,:])
        dx = xleft[1]-xleft[0]
        Bleft = squeeze(data['B1'][k, :, :, -ghost])

        ny, nz = shape(Bleft)
        
        vmin = Bleft.min() ; vmax = Bleft.max()
        
        w = rightlist[k]

        if size(w) > 0:
            print("meshblock ", k, ", level ", levels[k], " has (a) neighbour(s) to the right No ", w, " levels ", levels[w])
            print("left Y = ( ", yleft.min(), ", ", yleft.max(), ") ; Z = ( ", zleft.min(),", ", zleft.max(), ")")

            clf()
            fig, ax = subplots(1,2)
            plc = ax[0].pcolormesh(yleft,zleft,(Bleft), vmin = vmin , vmax = vmax )
            ax[0].set_xlim(yleft.min(), yleft.max())        ;            ax[0].set_ylim(zleft.min(), zleft.max())
            ax[0].set_xlabel(r'$Y$')   ;             ax[0].set_ylabel(r'$Z$')
            
            ax[0].set_title('X = '+str(xleft[-ghost-1]))
                        
            for kk in w:
                
                xright = squeeze(data['x1f'][kk,:])
                yright = squeeze(data['x2v'][kk,:])
                zright = squeeze(data['x3v'][kk,:])
                Bright = squeeze(data['B1'][kk, :, :, ghost])

                kkktr = 0
                
                ax[1].pcolormesh(yright,zright,(Bright), vmin = vmin , vmax = vmax)
                
                ax[1].plot([yright.min(), yright.max()], [zright.min(), zright.min()], 'g:')
                ax[1].plot([yright.min(), yright.max()], [zright.max(), zright.max()], 'g--')
                ax[1].plot([yright.min(), yright.min()], [zright.min(), zright.max()], 'g:')
                ax[1].plot([yright.max(), yright.max()], [zright.min(), zright.max()], 'g--')

                print(" right Y = ( ", yright.min(),", ", yright.max(), ") ; Z = ( ", zright.min(),", ", zright.max(), ")")
                if (levels[k] == levels[kk]) and (yleft.min() == yright.min()) and (zleft.min() == zright.min()):
                    # MFs should match exactly
                    ddB =  abs(Bleft-Bright).max()*dx / (abs(Bleft).max()+abs(Bright).max())
                    if ddB > 1e-10:
                        print("max |Delta B| = ", ddB)
                        print("same levels = ", levels[k])
                        plot_diff(Bright, Bleft, yright, zright, fname ='kleft{:03d}'.format(k)+'_{:03d}'.format(kk))
                        # ii = input('l')
                if (levels[kk] == (levels[k]+1)):
                    kkktr += 1
                    # and (yleft.min() == yright.min()) and (zleft.min() == zright.min()):
                    print("shapes: ", shape(Bleft), shape(Bright))
                    print("coord shapes: ", shape(yleft), shape(zleft), shape(yright), shape(zright))
                    print("left corner: ", yleft[2:(ny//2)], zleft[2:(nz//2)])
                    print("right corner: ", (yright[1:]+yright[:-1])[2:-1:2]/2., (zright[1:]+zright[:-1])[2:-1:2]/2.)
                    # we need to be sure that the coordinates match as well 
                    yav = (yright[1:]+yright[:-1])[2:-1:2]/2.
                    zav = (zright[1:]+zright[:-1])[2:-1:2]/2.
                    ycut = yleft[2:(ny//2)]
                    zcut = zleft[2:(nz//2)]
                    if (abs(yav-ycut).max() < dx) & (abs(zav-zcut).max() < dx):
                        Bav = (Bright[:-1, 1:] + Bright[:-1, :-1] + Bright[1:, 1:] + Bright[1:, :-1])/4.
                        Bav = Bav[2:-2:2, 2:-2:2]
                        Bcut = Bleft[2:(ny//2), 2:(nz//2)]
                        print("X match = ", xright[ghost], " = ", xleft[-ghost-1])
                        print("k = ", k, " -> kk = ", kk)
                        print("shapes: ", shape(Bav), shape(Bcut))
                        print("maximal field (right): ", +abs(Bright).max())
                        print("maximal field (left): ", +abs(Bleft).max())
                        print("max |Delta B| = ", abs(Bav-Bcut).max()*dx / (abs(Bleft).max()+abs(Bright).max()))
                        plot_diff(Bav, Bcut, ycut, zcut, fname ='kleft{:03d}'.format(k)+'_{:03d}'.format(kk))
                        if (abs(Bav-Bcut).max()*dx / (abs(Bleft).max()+abs(Bright).max()) > 1e-7):
                            ii = input('l+1')
                
            ax[1].set_xlabel(r'$Y$')   ;           ax[1].set_ylabel(r'$Z$')
            ax[1].set_xlim(yleft.min(), yleft.max())        ;            ax[1].set_ylim(zleft.min(), zleft.max())
            
            ax[1].set_title('X = '+str(xright[ghost]))

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(plc, cax=cbar_ax)
            # colorbar(plc)
            fig.set_size_inches(10., 5.)
            savefig('kleft{:03d}'.format(k)+'.png')
            #if kkktr > 0:
            #    ii = input('k')
            # print("X match = ", xleft[-ghost-1], "==", xright[ghost])
            

                        
# ffmpeg -f image2 -r 10 -pattern_type glob -i 'models/loopXneg/B*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p -b 8192k models/loopXneg.mp4

