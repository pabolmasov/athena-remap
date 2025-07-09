from numpy import *
from matplotlib import gridspec

import os
import sys
import glob
sys.path.append("vis/python")
# sys.path.append("/Users/pasha/athena/vis/python")
import athena_read

import h5py

# from scipy.optimize import root_scalar
# from scipy.integrate import simpson
# from scipy.integrate import cumulative_trapezoid as cumtrapz

from os.path import exists

ifplot = True
if(ifplot):
    import matplotlib
    from matplotlib.pyplot import *
    
cmap = 'plasma'

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
    
    vmin = quantile(q, 0.001)
    vmax = quantile(q, 0.999)
    
    clf()
    
    for k in arange(nlevels):
        x = data['x1v'][k,:]
        y = data['x2v'][k,:]
        z = data['x3v'][k,:]
        
        dz = z[1]-z[0]

        #print("z = ", z.min(), "..", z.max())
        if (z.max() > zslice) and (z.min() < zslice):
        
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

# reading 2d

def qplot2d(nfile, ddir = 'models/loop', qua = 'rho', prefix = 'Loop.prim', zslice = 0., domainnet = True, iflog = False):

    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = [qua], raw = False)

    time = data['Time']
    
    q = data[qua]

    x = data['x1v']
    y = data['x2v']

    m =  q.sum()*(x[1]-x[0])*(y[1]-y[0])
    
    if iflog:
        q = log10(q)
        
    vmin = quantile(q, 0.01)
    vmax = quantile(q, 0.99)
    
    clf()
    fig = figure()
    pc = pcolormesh(x, y, squeeze(q), vmin=vmin, vmax=vmax, cmap = cmap)
    print("q = ", q.min(), "..", q.max())

    xlabel('$X$')  ;  ylabel('$Y$')
    cb = colorbar(pc)
    if iflog:
        cb.set_label(r'$\log_{10}$'+qua)
    else:
        cb.set_label(qua)
    title(r'$t = '+str(round(time, 2))+'$')
    fig.set_size_inches(20.,3.0)
    savefig(ddir+'/'+qua+'{:05d}'.format(nfile)+'.png') #, DPI=500)

    return time, m

def Bplot2d(nfile, ddir = 'models/loop', prefix = 'Loop.prim', qua='press', iflog = True):

    filename = ddir+'/'+prefix+'.{:05d}'.format(nfile)+'.athdf'
    data = athena_read.athdf(filename, quantities = ['Bcc1', 'Bcc2', qua], raw = False)

    time = data['Time']
    
    q = data[qua]

    if iflog:
        q = log10(q)
        
    vmin = quantile(q, 0.01)
    vmax = quantile(q, 0.99)

    x = data['x1v'] ;    y = data['x2v']
    
    Bx = data['Bcc1']  ; By = data['Bcc2']

    print(shape(Bx))
    
    clf()
    fig, ax  = subplots()
    pc = pcolormesh(x, y, squeeze(q), vmin=vmin, vmax=vmax, cmap = 'viridis')
    xlabel('$X$')  ;  ylabel('$Y$')

    cb = colorbar(pc, ax = ax)
    
    streamplot(x, y, (squeeze(Bx)), (squeeze(By)), color = squeeze(By/Bx), cmap = 'Grays')
    cb1 = colorbar(ax = ax)
    
    if iflog:
        cb.set_label(r'$\log_{10}$'+qua)
    else:
        cb.set_label(qua)
    cb1.set_label(r'$B_y/B_x$')
        
    title(r'$t = '+str(round(time, 2))+'$')
    fig.set_size_inches(15.,8.0)
    savefig(ddir+'/Bstream{:05d}'.format(nfile)+'.png') #, DPI=500)
    
    
def movieq2d(narr, ddir='models/bowtest', qua = 'press', iflog = False, prefix = 'SSW.prim'):

    nnarr =size(narr)
    
    tar = zeros(nnarr) ;  mar = zeros(nnarr)
    
    for k in narr:
        ttmp, mtmp = qplot2d(k, ddir=ddir, prefix=prefix, qua=qua, iflog = iflog)
        tar[k] = ttmp ; mar[k] = mtmp

    clf()
    plot(tar, mar, 'k.')
    xlabel(r'$t$') ;  ylabel(r'$m$')
    savefig('marcurve.png')
