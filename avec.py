# restores 3D vector potential from magnetic field components. Resulting grid is uniform.

from numpy import *
from matplotlib import gridspec

from threading import Thread

# from numba import njit
# from numba.openmp import openmp_context as openmp
# from numba.openmp import omp_get_thread_num, omp_get_num_threads

import os
import sys
import glob
sys.path.append("../vis/python")
# sys.path.append("/Users/pasha/athena/vis/python")
import athena_read

import h5py

from scipy.optimize import root_scalar
from scipy.integrate import simpson
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d, griddata

from ast import literal_eval

from os.path import exists

ifplot = True
if(ifplot):
    import matplotlib
    from matplotlib.pyplot import *

# omp_get_num_threads()
interplots = True # intermediate output diagnostic plots

def ispoweroftwo(n):
    # bit manipulations trick from https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
    return n & (n-1) == 0

# one Avec component treated by one thread. The routine does all the relaxation iterations for one component of A
def Poi_iter(a, j, inds, dx, zoom, tol, omega, kzr, ghostlayers, thnum):
        
    xind3, yind3, zind3 = inds
    
    print("shapes:", shape(xind3), shape(a))
    
    a1 = copy(a)
    da = copy(a)
    
    daa = abs(a).max()
    aa = abs(a).max()
    kiter = 0
    
    while (daa > (tol * aa)) and (kiter < 10000):
        
        da[::zoom,::zoom,::zoom] = (roll(a1[::zoom,::zoom,::zoom], 1, axis=0) + roll(a1[::zoom,::zoom,::zoom], -1, axis=0) + roll(a1[::zoom,::zoom,::zoom],1, axis=1) + roll(a1[::zoom,::zoom,::zoom], -1, axis=1) + roll(a1[::zoom,::zoom,::zoom],1, axis=2) + roll(a1[::zoom,::zoom,::zoom], -1, axis=2)) + j[::zoom,::zoom,::zoom] * dx**2 # requires dx = dy = dz

        
        if (ghostlayers> 0):
            da[-ghostlayers:,:,:] = 0. ;     da[:ghostlayers,:,:] = 0.
            da[:,-ghostlayers:,:] = 0. ;     da[:,:ghostlayers,:] = 0.
            da[:,:,-ghostlayers:] = 0. ;     da[:,:,:ghostlayers] = 0.
        
        # print("thread number ", thnum, ": max a = ", a.max())
        # checkerboard pattern:
        if kiter%2==0:
            w = (xind3+yind3+zind3)%(2*zoom)==0
        else:
            w = (xind3+yind3+zind3)%(2*zoom)==zoom

        daa = abs(da[w]/6.-a1[w]).max()            
            
        a1[w] = (1.-omega) * a1[w] + omega * da[w]/6.
        
        aa = abs(a1[w]-a1[w].mean()).max()
        
        if (daa < (tol * aa*zoom)) and (zoom > 1):
            print("thread number ", thnum, ": max a = ", a1.max())
            zoom //= 2
            w0 = (xind3+yind3+zind3)%(2*zoom)==0
            w1 = (xind3+yind3+zind3)%(2*zoom)==zoom
            a1[w1] = a1[w0]
            
        kiter += 1
        if kiter%100==0:
            print("thread number ", thnum, ":  kiter = ", kiter, '; da = ', daa, ' / ', aa, ' = ', daa/aa)
            if interplots:
                clf()
                fig = figure()
                print(shape(a1))
                pcolormesh(squeeze(a1[kzr, ::zoom, ::zoom]), vmin = quantile(a1[kzr, ::zoom, ::zoom], 0.01), vmax = quantile(a1[kzr, ::zoom, ::zoom], 0.99))
                colorbar()
                # xlim(-100., 100.) ; ylim(-100., 100.)
                xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$k = '+str(kiter)+'$', fontsize=14)
                fig.set_size_inches(12.,12.)
                savefig('A{:02d}'.format(thnum+1)+'map{:05d}.png'.format(kiter))
                close()

    a[:] = a1[:]
    
def Avec(nfile, ddir = 'loopX', prefix = 'Loop', alias = 2, xrange = None, yrange = None, zrange = None, omega = 1.0, tol = 1e-6, tracercutoff = 0., rhocutoff = 1e-6, ifasc = False, massboost = 1.0, newcoords = False, nxnew = 128, nynew = 128, nznew = 128, ghostlayers = 0, zslice = 0., zoomin = 1, enforceraw = False, interplots = True):
    '''
    restoring the vector potential using the set of primitive values
    nfile is the number of the input hdf5 file
    alias = integer uses only part of the datapoints in each direction
    ...
    tracercutoff: between 0 and 1, cut-off value for r0
    ...
    ghostlayers = integer: if > 0, smooths the data iteratively and zeroes the border layers (ghostlayers) 
    zoomin = power of 2:  solves the Poisson equation on a coarser grid first
    enforceraw makes the reader retain the meshblocks of the original file. Useful if there is not enough memory for reading the regular grid; should work only with newcoords
    '''

    if ispoweroftwo(zoomin):
        zoom = zoomin
    else:
        print("zoom factor = ", zoomin, " not a factor of 2")
        exit(1)    
    
    filename = ddir+'/'+prefix+'.prim.'+'{:05d}'.format(nfile)+'.athdf'

    if (tracercutoff > 0.) or (massboost > 1.0):
        data = athena_read.athdf(filename, quantities = ['Bcc1', 'Bcc2', 'Bcc3', 'r0', 'rho'], raw = enforceraw)
    else:
        data = athena_read.athdf(filename, quantities = ['Bcc1', 'Bcc2', 'Bcc3', 'rho'], raw = enforceraw)

    #  Bdata = athena_read.athdf(bfilename, quantities = ['B1', 'B2', 'B3'], raw = False, num_ghost=2) # does not handle ghost zone

            
    print("data read")
    if not(enforceraw):
        B1 = data['Bcc1']  ; B2 = data['Bcc2'] ; B3 = data['Bcc3']
    
    x = data['x1v']  ;   y = data['x2v'] ;   z = data['x3v'] 
    xf = data['x1f']  ;   yf = data['x2f'] ;   zf = data['x3f'] 
    
    if newcoords:
        # the mesh of new cell centres
        xnew = arange(nxnew) / double(nxnew-1) * (xrange[1]-xrange[0]) + xrange[0]
        ynew = arange(nynew) / double(nynew-1) * (yrange[1]-yrange[0]) + yrange[0]
        znew = arange(nznew) / double(nznew-1) * (zrange[1]-zrange[0]) + zrange[0]

        # cell faces
        xnewf = (arange(nxnew+1)-0.5) / double(nxnew-1) * (xrange[1]-xrange[0]) + xrange[0]
        ynewf = (arange(nynew+1)-0.5) / double(nynew-1) * (yrange[1]-yrange[0]) + yrange[0]
        znewf = (arange(nznew+1)-0.5) / double(nznew-1) * (zrange[1]-zrange[0]) + zrange[0]
    
        z3newf3, y3new3, x3new3 = meshgrid(znewf, ynew, xnew, indexing = 'ij') # mesh for Bz
        z3new2, y3newf2, x3new2 = meshgrid(znew, ynewf, xnew, indexing = 'ij') # mesh for By
        z3new1, y3new1, x3newf1 = meshgrid(znew, ynew, xnewf, indexing = 'ij') # mesh for Bx
        z3new, y3new, x3new = meshgrid(znew, ynew, xnew, indexing = 'ij') # mesh for hydro
        
        dxnew = xnew[1]-xnew[0]  ;  dynew = ynew[1]-ynew[0]  ;  dznew = znew[1]-znew[0]
        
        print(shape(x))
        print("resolution: ", (x[:,1]-x[:,0]).min(), " -> ", dxnew)
        
        if enforceraw:
            # then we need to create all the regular arrays by hand
            rhonew = zeros([nznew, nynew, nxnew])
            r0new = zeros([nznew, nynew, nxnew])
            B1new = zeros([nznew, nynew, nxnew+1])
            B2new = zeros([nznew, nynew+1, nxnew])
            B3new = zeros([nznew+1, nynew, nxnew])
            
            # now we need a remapping routine, one meshblock at a time
            levels = data['Levels']
            # print("maximal level is ", levels.max())
            nlevels=size(levels)
            # print(nlevels, " blocks in total")
            
            for k in arange(nlevels):
                x = data['x1v'][k, :]  ;   y = data['x2v'][k, :]  ;   z = data['x3v'][k, :]  
                xf = data['x1f'][k, :]   ;   yf = data['x2f'][k, :]  ;   zf = data['x3f'][k, :]  
                dx = x[1]-x[0] ; dy = y[1]-y[0] ; dz = z[1]-z[0]
                z3, y3, x3 = meshgrid(z, y, x, indexing='ij')
                z3f, y3f, x3f = meshgrid(zf, yf, xf, indexing='ij')
                rho = data['rho'][k,:] ; r0 = data['r0'][k,:]
                B1 = data['Bcc1'][k,:] ; B2 = data['Bcc2'][k, :] ; B3 = data['Bcc3'][k,:]
                #if (tracercutoff > 0.) or (massboost > 1.0):
                #    scalefactor = 1. + (massboost-1.) * r0
                    # print("scalefactor range = ", scalefactor.min(), scalefactor.max())
                    # ii = input('S')
                    # print("|Bx| <= ", abs(B1).max())
                #    B1 *= sqrt(scalefactor)  ;       B2 *= sqrt(scalefactor)  ;    B3 *= sqrt(scalefactor)
                    # we need to amplify the fields or the currents? 
                    # print("|Bx| <= ", abs(B1).max())
                    
                w = (x3new >= (x3.min()-dx-dxnew)) & (x3new <= (x3.max()+dx+dxnew)) &  (y3new >= (y3.min()-dy-dynew)) & (y3new <= (y3.max()+dy+dynew)) &  (z3new >= (z3.min()-dz-dznew)) & (z3new <= (z3.max()+dz+dznew))
                if w.sum() > 0:
                    rhonew[w] = griddata((z3.flatten(), y3.flatten(), x3.flatten()), rho.flatten(), (z3new[w], y3new[w], x3new[w]), method = 'nearest') 
                    r0new[w] = griddata((z3.flatten(), y3.flatten(), x3.flatten()), r0.flatten(), (z3new[w], y3new[w], x3new[w]), method = 'nearest')
                # note! mapping Bcc to face centres here!                
                wf1 = (x3newf1 >= (x3.min()-dx-dxnew)) & (x3newf1 <= (x3.max()+dx+dxnew)) &  (y3new1 >= (y3.min()-dy-dynew)) & (y3new1 <= (y3.max()+dy+dynew)) &  (z3new1 >= (z3.min()-dz-dznew)) & (z3new1 <= (z3.max()+dz+dznew))
                wf2 = (x3new2 >= (x3.min()-dx-dxnew)) & (x3new2 <= (x3.max()+dx+dxnew)) &  (y3newf2 >= (y3.min()-dy-dynew)) & (y3newf2 <= (y3.max()+dy+dynew)) &  (z3new2 >= (z3.min()-dz-dznew)) & (z3new2 <= (z3.max()+dz+dznew))
                wf3 = (x3new3 >= (x3.min()-dx-dxnew)) & (x3new3 <= (x3.max()+dx+dxnew)) &  (y3new3 >= (y3.min()-dy-dynew)) & (y3new3 <= (y3.max()+dy+dynew)) &  (z3newf3 >= (z3.min()-dz-dznew)) & (z3newf3 <= (z3.max()+dz+dznew))
                # print(shape(B1new), shape(wf1))
                if wf1.sum() > 0:
                    B1new[wf1] = griddata((z3.flatten(), y3.flatten(), x3.flatten()), B1.flatten(), (z3new1[wf1], y3new1[wf1], x3newf1[wf1]), method = 'nearest') 
                if wf2.sum() > 0:
                    B2new[wf2] = griddata((z3.flatten(), y3.flatten(), x3.flatten()), B2.flatten(), (z3new2[wf2], y3newf2[wf2], x3new2[wf2]), method = 'nearest') 
                if wf3.sum():
                    B3new[wf3] = griddata((z3.flatten(), y3.flatten(), x3.flatten()), B3.flatten(), (z3newf3[wf3], y3new3[wf3], x3new3[wf3]), method = 'nearest') 
            
            kz = abs(znew-zslice).argmin()  
            
            clf()
            pc1 = pcolormesh(xnew, ynew, squeeze(B1new[kz, :, 1:]), vmin = B1new.min(), vmax = B1new.max())
            colorbar(pc1)
            savefig('B1map.png')
            clf()
            pc1 = pcolormesh(xnew, ynew, squeeze(B2new[kz, 1:, :]), vmin = B2new.min(), vmax = B2new.max())
            colorbar(pc1)
            savefig('B2map.png')
            clf()
            pc1 = pcolormesh(xnew, ynew, squeeze(B3new[kz, :, :]), vmin = B3new.min(), vmax = B3new.max())
            colorbar(pc1)
            savefig('B3map.png')
            
            print('HD and fields remapped')
            j1new = zeros([nznew+1, nynew+1, nxnew])
            j2new = zeros([nznew+1, nynew, nxnew+1])
            j3new = zeros([nznew, nynew+1, nxnew+1])

            dx = dxnew ; dy=dynew ; dz = dznew
            
            # nz+1, ny-1, nz+1 ; nz-1, ny+1, nx
            j1new[:,1:-1,:] = (B3new[:,1:,:] - B3new[:,:-1,:])/dynew
            j1new[1:-1,:,:] -= (B2new[1:,:,:]-B2new[:-1,:,:])/dznew 
            # nz-1, ny, nx+1 ; nz+1, ny, nx-1 
            j2new[1:-1,:,:] = (B1new[1:,:,:] - B1new[:-1,:,:])/dznew 
            j2new[:,:,1:-1] -= (B3new[:,:,1:]-B3new[:,:,:-1])/dxnew 
            # nz+1, ny, nx-1 ; nx+1, ny-1, nz
            j3new[:,:,1:-1]  = (B2new[:,:,1:] - B2new[:,:,:-1])/dxnew 
            j3new[:,1:-1,:] -= (B1new[:,1:,:]-B1new[:,:-1,:])/dynew 
    else:
        # read cooked data
        xnew = x ; ynew = y ; znew = z
        if (tracercutoff > 0.) or (massboost > 1.0):
            r0new = data['r0']
        rhonew = data['rho']
        dxnew = x[1]-x[0]  ;  dynew = y[1]-y[0]  ;   dznew = z[1]-z[0]
        dx = x[1]-x[0]  ;  dy = y[1]-y[0]  ;   dz = z[1]-z[0]
        nxnew = size(x) ; nynew = size(y) ; nznew = size(z)

        kz = abs(znew-zslice).argmin()  

        j1new = zeros([nznew+1, nynew+1, nxnew])
        j2new = zeros([nznew+1, nynew, nxnew+1])
        j3new = zeros([nznew, nynew+1, nxnew+1])
        
        # nz+1, ny-1, nz+1 ; nz-1, ny+1, nx
        j1new[1:-1,1:-1,:] = ((B3[1:,1:,:] - B3[1:,:-1,:]) + (B3[:-1,1:,:] - B3[:-1,:-1,:]))/dynew/2.
        j1new[1:-1,1:-1,:] -= ((B2[1:,1:,:]-B2[:-1,1:,:]) + (B2[1:,:-1,:]-B2[:-1,:-1,:]))/dznew/2. 
        # nz-1, ny, nx+1 ; nz+1, ny, nx-1 
        j2new[1:-1,:,1:-1] = ((B1[1:,:,1:] - B1[:-1,:,1:]) + (B1[1:,:,:-1] - B1[:-1,:,:-1]))/dznew/2. 
        j2new[1:-1,:,1:-1] -= ((B3[1:,:,1:]-B3[1:,:,:-1]) + (B3[:-1,:,1:]-B3[:-1,:,:-1]))/dxnew /2.
        # nz+1, ny, nx-1 ; nx+1, ny-1, nz
        j3new[:,1:-1,1:-1]  = ((B2[:,1:,1:] - B2[:,1:,:-1]) + (B2[:,:-1,1:] - B2[:,:-1,:-1]))/dxnew /2.
        j3new[:,1:-1,1:-1] -= ((B1[:,1:,1:]-B1[:,:-1,1:]) + (B1[:,1:,:-1]-B1[:,:-1,:-1]))/dynew/2.         
        
        # if newcoords and enforceraw:
    jscale = sqrt((j1new[1:,1:,:]**2+j2new[1:,:,1:]**2+j3new[:,1:,1:]**2).mean())
    j1new /= jscale ; j2new /= jscale ; j3new /= jscale # normalizing currents; Avec will be scaled back before output

    if massboost > 1.0:
        scalefactor = 1. + (massboost-1.) * r0new
        j1new[1:,1:,:] *= sqrt(scalefactor)
        j2new[1:,:,1:] *= sqrt(scalefactor)
        j3new[:,1:,1:] *= sqrt(scalefactor)

    clf()
    fig = figure()
    print(shape(xnew), shape(ynew), shape(j3new))
    # pcolormesh(x, y, (squeeze(abs(j1).max(axis=0))))
    pcolormesh(xnew, ynew, j3new[kz,1:,1:])
    colorbar()
    # xlim(-100., 100.) ; ylim(-100., 100.)
    xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_x$', fontsize=14)
    fig.set_size_inches(12.,12.)
    savefig('j3map_old.png')
    #        exit(0)
    '''
    if newcoords and not(enforceraw):
        print("j scale = ", jscale)
        print("z3 shape: ", shape(z3new))
        print("size of the old box: X= ", x.min(), x.max(), "; Y = ", y.min(), y.max(), "; Z = ", z.min(), z.max())
        
        if interplots:
            clf()
            fig = figure()
            # pcolormesh(x, y, (squeeze(abs(j1).max(axis=0))))
            pcolormesh(x, y, j1[kz,:,:])
            colorbar()
            # xlim(-100., 100.) ; ylim(-100., 100.)
            xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_x$', fontsize=14)
            fig.set_size_inches(12.,12.)
            savefig('j1map_old.png')
            close()
            print("max jx  = ", abs(j1).max())
            clf()
            fig = figure()
            # pcolormesh(x, y, (squeeze(abs(j1).max(axis=0))))
            pcolormesh(x, y, j2[kz,:,:])
            colorbar()
            # xlim(-100., 100.) ; ylim(-100., 100.)
            xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_y$', fontsize=14)
            fig.set_size_inches(12.,12.)
            savefig('j2map_old.png')
            close()
            print("max jy  = ", abs(j2).max())
            clf()
            fig = figure()
            # pcolormesh(x, y, (squeeze(abs(j3).max(axis=0))))
            pcolormesh(x, y, j3[kz,:,:])
            colorbar()
            # xlim(-100., 100.) ; ylim(-100., 100.)
            xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_z$', fontsize=14)
            fig.set_size_inches(12.,12.)
            savefig('j3map_old.png')
            close()
            print("max jz  = ", abs(j3).max())
            # ii = input('J')
    
        j1 = reshape(griddata((z3.flatten(), y3.flatten(), x3.flatten()), j1.flatten(), (z3new.flatten(), y3new.flatten(), x3new.flatten()), method='nearest', fill_value = 0.), shape(x3new))
        j2 = reshape(griddata((z3.flatten(), y3.flatten(), x3.flatten()), j2.flatten(), (z3new.flatten(), y3new.flatten(), x3new.flatten()), method='nearest', fill_value = 0.), shape(x3new))
        j3 = reshape(griddata((z3.flatten(), y3.flatten(), x3.flatten()), j3.flatten(), (z3new.flatten(), y3new.flatten(), x3new.flatten()), method='nearest', fill_value = 0.), shape(x3new))
    
        x = xnew ; y = ynew ; z = znew
        
        #  j1 = reshape(j1new, shape(x3new)) ; j2 = reshape(j2new, shape(y3new)) ; j3 = reshape(j3new, shape(z3new))
        
        dx = x[1] - x[0]  ; dy = y[1] - y[0]  ; dz = z[1] - z[0]
        print("new coordinates")
        print("dx, dy, dz = ", dx, dy, dz)
    ''' 
    
    kzr = abs(z-zslice).argmin()    

    dx = dxnew
    print("dx = ", dx)
    
    x = xnew ; y = ynew ; z = znew

    j1 = copy(j1new) ; j2 = copy(j2new) ; j3 = copy(j3new)
    
    # make a smoothing routine
    nsmooth = ghostlayers
    
    for k in arange(nsmooth):
        #print("max jx  = ", abs(j1new).max())
        #print("max jy  = ", abs(j2new).max())
        #print("max jz  = ", abs(j3new).max())
        j1 = j1new*0. ; j2 = j2new*0. ; j3 = j3new*0.
        for shiftx in arange(3)-1:
            for shifty in arange(3)-1:
                for shiftz in arange(3)-1:
                    j1 += roll(roll(roll(j1new, shiftz, axis=0), shifty, axis=1), shiftx, axis=2)
                    j2 += roll(roll(roll(j2new, shiftz, axis=0), shifty, axis=1), shiftx, axis=2)
                    j3 += roll(roll(roll(j3new, shiftz, axis=0), shifty, axis=1), shiftx, axis=2)
        j1 /= 27. ; j2 /= 27. ; j3 /= 27.
        j1new[:] = j1[:] ; j2new[:] = j2[:] ; j3new[:] = j3[:]
        #print("max jx  = ", abs(j1new).max())
        #print("max jy  = ", abs(j2new).max())
        #print("max jz  = ", abs(j3new).max())
        # ii = input("jx")
        j1new[:ghostlayers,:,:] = 0. ;   j2new[:ghostlayers,:,:] = 0. ;  j2new[:ghostlayers,:,:] = 0.
        j1new[-ghostlayers:,:,:] = 0. ;   j2new[-ghostlayers:,:,:] = 0. ;    j3new[-ghostlayers:,:,:] = 0.
        j1new[:,:ghostlayers,:] = 0. ;   j2new[:,:ghostlayers,:] = 0. ;  j2new[:,:ghostlayers,:] = 0.
        j1new[:,-ghostlayers:,:] = 0. ;   j2new[:,-ghostlayers:,:] = 0. ;    j3new[:,-ghostlayers:,:] = 0.
        j1new[:,:,:ghostlayers] = 0. ;   j2new[:,:,:ghostlayers] = 0. ;  j2new[:,:,:ghostlayers] = 0.
        j1new[:,:,-ghostlayers:] = 0. ;   j2new[:,:,-ghostlayers:] = 0. ;    j3new[:,:,-ghostlayers:] = 0.

    if tracercutoff > 0.:
        j1new[1:,1:,:] *= (r0new > tracercutoff)
        j2new[1:,:,1:] *= (r0new > tracercutoff)
        j3new[:,1:,1:] *= (r0new > tracercutoff)                    
        
    nx = size(x) ; ny = size(y) ; nz = size(z)
    
    print("dimensions ", nx, "X", ny, "X", nz)

    print("B1 dimensions", shape(B1))

    # ii = input("B")    
    # j1 = zeros([nz+1, ny+1, nx]) ; j2 = zeros([nz+1, ny, nx+1]) ; j3 = zeros([nz, ny+1, nx+1]) 
    A1 = copy(j1) * 0.  ; A2 = copy(j2) * 0.  ; A3 = copy(j3) * 0.
    da1 = copy(j1) * 0.  ; da2 = copy(j2) * 0.  ; da3 = copy(j3) * 0.
    
    # A1-3 seed shape smart guess:
    
    print(shape(j3), shape(A3), shape(xnew))

    # a crude guess for Avec:
    A1[1:, 1:, :] = j3[:, 1:, 1:] * dxnew**2 
    A2[1:, :, 1:] = j1[1:, 1:, :] * dxnew**2
    A3[:, 1:, 1:] = j2[1:, :, 1:] * dxnew**2
    
    
    # j1 = (B3[:, 1:, :] - B3[:, :-1, :])/dy -  (B2[1:, :, :] - B2[:-1, :, :])/dz
    # j2 = (B1[1:, :, :] - B1[:-1, :, :])/dz -  (B3[:, :, 1:] - B3[:, :, :-1])/dx
    # j3 = (B2[:, :, 1:] - B2[:, :, :-1])/dx -  (B1[:, 1:, :] - B1[:, :-1, :])/dy

    # tol = 1e-6
    da = tol*100.
    aa = 0.0
    
    kiter = 0

    if interplots:
        clf()
        fig = figure()
        print(shape(x), shape(y), shape(j1))
        # pcolormesh(x, y, (squeeze(abs(j1).max(axis=0))))
        pcolormesh(xnew, ynew, j1new[kz,1:,:])
        colorbar()
        # xlim(-100., 100.) ; ylim(-100., 100.)
        xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_x$', fontsize=14)
        fig.set_size_inches(12.,12.)
        savefig('j1map.png')
        close()
        print("max jx  = ", abs(j1).max())
        clf()
        fig = figure()
        # pcolormesh(x, y, (squeeze(abs(j1).max(axis=0))))
        pcolormesh(xnew, ynew, j2new[kz,:,1:])
        colorbar()
        # xlim(-100., 100.) ; ylim(-100., 100.)
        xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_y$', fontsize=14)
        fig.set_size_inches(12.,12.)
        savefig('j2map.png')
        close()
        print("max jy  = ", abs(j2).max())
        clf()
        fig = figure()
        # pcolormesh(x, y, (squeeze(abs(j3).max(axis=0))))
        pcolormesh(x, y, j3new[kz,1:,1:])
        colorbar()
        # xlim(-100., 100.) ; ylim(-100., 100.)
        xlabel(r'$x$', fontsize=14) ; ylabel(r'$y$', fontsize=14) ; title(r'$j_z$', fontsize=14)
        fig.set_size_inches(12.,12.)
        savefig('j3map.png')
        close()
        print("max jz  = ", abs(j3).max())
        # ii = input('J')    
    
    # let us try to thread this! TODO: make threading optional
    A1_0 = copy(A1) ; A2_0 = copy(A2)  ; A3_0 = copy(A3)
    nthreads = 3
    xind = arange(shape(A1)[2]) ; yind = arange(shape(A1)[1]) ; zind = arange(shape(A1)[0])
    zind3, yind3, xind3 = meshgrid(zind, yind, xind, indexing = 'ij')
    t1 = Thread(target = Poi_iter, args = (A1, j1new, [xind3, yind3, zind3], dx, zoom, tol, omega, kz, ghostlayers, 0))
    xind = arange(shape(A2)[2]) ; yind = arange(shape(A2)[1]) ; zind = arange(shape(A2)[0])
    zind3, yind3, xind3 = meshgrid(zind, yind, xind, indexing = 'ij')
    t2 = Thread(target = Poi_iter, args = (A2, j2new, [xind3, yind3, zind3], dy, zoom, tol, omega, kz, ghostlayers, 1))
    xind = arange(shape(A3)[2]) ; yind = arange(shape(A3)[1]) ; zind = arange(shape(A3)[0])
    zind3, yind3, xind3 = meshgrid(zind, yind, xind, indexing = 'ij')
    t3 = Thread(target = Poi_iter, args = (A3, j3new, [xind3, yind3, zind3], dz, zoom, tol, omega, kz, ghostlayers, 2))
    
    t1.start() ;     t2.start() ;   t3.start()
    
    t1.join() ; t2.join() ; t3.join()
    
    print("A changed after solving")
    print((A1-A1_0).std())
    print((A2-A2_0).std())
    print((A3-A3_0).std())
    #    ii = input("?")            
    
    #plot the A map
    if interplots:
        print("making plots")
        clf()
        pcolormesh(squeeze(A3[kz, :, :]))
        xlabel(r'$x$') ; ylabel(r'$y$') ;  title(r'$k = '+str(kiter)+'$')
        fig.set_size_inches(12.,12.)
        savefig('A3map.png')
    
    # if newcoords:
    A1 *= jscale ; A2 *= jscale ; A3 *= jscale

    if (ghostlayers> 0):
        A1[-ghostlayers:,:,:] = 0. ;     A1[:ghostlayers,:,:] = 0.
        A1[:,-ghostlayers:,:] = 0. ;     A1[:,:ghostlayers,:] = 0.
        A1[:,:,-ghostlayers:] = 0. ;     A1[:,:,:ghostlayers] = 0.
        A2[-ghostlayers:,:,:] = 0. ;     A2[:ghostlayers,:,:] = 0.
        A2[:,-ghostlayers:,:] = 0. ;     A2[:,:ghostlayers,:] = 0.
        A2[:,:,-ghostlayers:] = 0. ;     A2[:,:,:ghostlayers] = 0.
        A3[-ghostlayers:,:,:] = 0. ;     A3[:ghostlayers,:,:] = 0.
        A3[:,-ghostlayers:,:] = 0. ;     A3[:,:ghostlayers,:] = 0.
        A3[:,:,-ghostlayers:] = 0. ;     A3[:,:,:ghostlayers] = 0.
        
    
    # ASCII output
    if ifasc:
        print("ASCII outputs to "+filename+'_A1,2,3.dat')
        Aout1 = open(filename+'_A1.dat', 'w')
        Aout2 = open(filename+'_A2.dat', 'w')
        Aout3 = open(filename+'_A3.dat', 'w')
        Aout1.write(str(nx)+' '+str(ny)+' '+str(nz)+'\n')
        Aout1.write(' '.join(x.astype(str))+'\n')
        Aout1.write(' '.join(y.astype(str))+'\n')
        Aout1.write(' '.join(z.astype(str))+'\n')
        Aout2.write(str(nx)+' '+str(ny)+' '+str(nz)+'\n')
        Aout2.write(' '.join(x.astype(str))+'\n')
        Aout2.write(' '.join(y.astype(str))+'\n')
        Aout2.write(' '.join(z.astype(str))+'\n')
        Aout3.write(str(nx)+' '+str(ny)+' '+str(nz)+'\n')
        Aout3.write(' '.join(x.astype(str))+'\n')
        Aout3.write(' '.join(y.astype(str))+'\n')
        Aout3.write(' '.join(z.astype(str))+'\n')
        
        for i in arange(nx):
            for j in arange(ny):
                # print(' '.join(A3[:,j,i].astype(str))+'\n')
                Aout1.write(' '.join(A1[:,j,i].astype(str))+'\n') # all zs for given x and y
                Aout2.write(' '.join(A2[:,j,i].astype(str))+'\n') # all zs for given x and y
                Aout3.write(' '.join(A3[:,j,i].astype(str))+'\n') # all zs for given x and y
                Aout1.flush()   ; Aout2.flush()  ;  Aout3.flush()
                Aout1.close()   ; Aout2.close()  ; Aout3.close()
                
    hname = filename + '_A.hdf5'
    hfile = h5py.File(hname, "w")
    # glo = hfile.create_group("globals")
    hfile.attrs["nx"] = nx
    hfile.attrs["ny"] = ny
    hfile.attrs["nz"] = nz
    
    # coords = hfile.create_group("coords")
    hfile.create_dataset("X", data = x)
    hfile.create_dataset("Y", data = y)
    hfile.create_dataset("Z", data = z)
    
    # A = hfile.create_group("Avec")
    hfile.create_dataset("A1", data=A1[:-1,:-1,:])
    hfile.create_dataset("A2", data=A2[:-1,:,:-1])
    hfile.create_dataset("A3", data=A3[:,:-1,:-1])

    hfile.flush()
    hfile.close()
    
        
    # magnetic field test
    B1r = (A3[:,1:,:]-A3[:,:-1,:])/dynew - (A2[1:,:,:]-A2[:-1,:,:])/dznew
    B2r = (A1[1:,:,:]-A1[:-1,:,:])/dznew - (A3[:,:,1:]-A3[:,:,:-1])/dxnew
    B3r = (A2[:,:,1:]-A2[:,:,:-1])/dxnew - (A1[:,1:,:]-A1[:,:-1,:])/dynew
    #B1r = (roll(A3, 1, axis = 1) - roll(A3, -1, axis = 1))/dy - (roll(A2, 1, axis=0)-roll(A2, -1, axis=0))/dz 
    #B2r = (roll(A1, 1, axis = 0) - roll(A1, -1, axis = 0))/dz - (roll(A3, 1, axis=2)-roll(A3, -1, axis=2))/dx 
    #B3r = (roll(A2, 1, axis = 2) - roll(A2, -1, axis = 2))/dx - (roll(A1, 1, axis=1)-roll(A1, -1, axis=1))/dy 
    print(shape(B1r), shape(B2r), shape(B3r))
    
    
    # B1r /= 2. ; B2r /= 2. ; B3r /= 2.
    
    '''
    if newcoords:
        B1rint = reshape(griddata((x3new.flatten(), y3new.flatten(), z3new.flatten()), B1r.flatten(), (x3.flatten(), y3.flatten(), z3.flatten()), method = 'nearest'), shape(B1))
        B2rint = reshape(griddata((x3new.flatten(), y3new.flatten(), z3new.flatten()), B2r.flatten(), (x3.flatten(), y3.flatten(), z3.flatten()), method = 'nearest'), shape(B2))
        B3rint = reshape(griddata((x3new.flatten(), y3new.flatten(), z3new.flatten()), B3r.flatten(), (x3.flatten(), y3.flatten(), z3.flatten()), method = 'nearest'), shape(B3)) 
        print("MF reshaped")
    else:
        B1rint = B1r
        B2rint = B2r
        B3rint = B3r
    '''

    if newcoords:
        db1 = abs(B1new-B1r).max()
        db2 = abs(B2new-B2r).max()
        db3 = abs(B3new-B3r).max()
        print("MF differences: |dB1| <= ", db1, " / ", abs(B1new).max(), ", |dB2| <= ", db2, " / ", abs(B2new).max(), ", |dB3| <= ", db3, " / ", abs(B3new).max())
    else:
        db1 = abs(B1-B1r[:,:,1:]).max()
        db2 = abs(B2-B2r[:,1:,:]).max()
        db3 = abs(B3-B3r[1:,:,:]).max()
        print("MF differences: |dB1| <= ", db1, " / ", abs(B1).max(), ", |dB2| <= ", db2, " / ", abs(B2).max(), ", |dB3| <= ", db3, " / ", abs(B3).max())
    
    # ii = input()

    if newcoords:
        if (abs(B1new).max() > 0.):
            lB1 = polyfit(B1new.flatten(), B1r.flatten(), deg=1)
        else:
            lB1 = [0.,0.]
        lB2 = polyfit(B2new.flatten(), B2r.flatten(), deg=1)
        lB3 = polyfit(B3new.flatten(), B3r.flatten(), deg=1)

        if interplots:
            clf()
            fig, ax = subplots(1,3)
            ax[0].plot(B1new.flatten(), B1r.flatten(), 'k,')
            ax[0].plot([B1new.min(), B1new.max()], [lB1[0]*B1new.min()+lB1[1], lB1[0]*B1new.max()+lB1[1]], 'r-')
            ax[0].set_xlabel(r'$B_{\rm x, \ old}$', fontsize=14) ; ax[0].set_ylabel(r'$B_{\rm x, \ new}$', fontsize=14) 
            ax[1].plot(B2new.flatten(), B2r.flatten(), 'k,')
            ax[1].plot([B2new.min(), B2new.max()], [lB2[0]*B2new.min()+lB2[1], lB2[0]*B2new.max()+lB2[1]], 'r-')
            ax[1].set_xlabel(r'$B_{\rm y, \ old}$', fontsize=14) ; ax[1].set_ylabel(r'$B_{\rm y, \ new}$', fontsize=14) 
            ax[2].plot(B3new.flatten(), B3r.flatten(), 'k,')
            ax[2].plot([B3new.min(), B3new.max()], [lB3[0]*B3new.min()+lB3[1], lB3[0]*B3new.max()+lB3[1]], 'r-')
            ax[2].set_xlabel(r'$B_{\rm z, \ old}$', fontsize=14) ; ax[2].set_ylabel(r'$B_{\rm z, \ new}$', fontsize=14) 
            fig.set_size_inches(15.,6.)
            savefig('Bscale.png')
            close()
    
    
            print("B1 ratio: ", lB1[0])
            print("B2 ratio: ", lB2[0])
            print("B3 ratio: ", lB3[0])
    
    
    if interplots:
        clf()
        fig, ax = subplots(2,1)
        if newcoords:
            pc1 = ax[0].pcolormesh(xnew, ynew, squeeze(B1new[kz, :, 1:]), vmin = B1new.min(), vmax = B1new.max())
        else:
            pc1 = ax[0].pcolormesh(xnew, ynew, squeeze(B1[kz, :, :]), vmin = B1.min(), vmax = B1.max())
            
        colorbar(pc1, ax = ax[0])
        if newcoords:
            pc2 = ax[1].pcolormesh(xnew, ynew, squeeze(B1r[kz, :, 1:]),  vmin = B1new.min(), vmax = B1new.max())
        else:
            pc2 = ax[1].pcolormesh(xnew, ynew, squeeze(B1r[kz, :, 1:]), vmin = B1.min(), vmax = B1.max())
        
        #ax[0].set_xlim(-100., 100.) ; ax[0].set_ylim(-100., 100.)
        # ax[1].set_xlim(-100., 100.) ; ax[1].set_ylim(-100., 100.)
        colorbar(pc2, ax = ax[1])
        ax[0].set_xlabel(r'$x$', fontsize=14) ; ax[0].set_ylabel(r'$y$', fontsize=14) 
        ax[1].set_xlabel(r'$x$', fontsize=14) ; ax[1].set_ylabel(r'$y$', fontsize=14) 
        suptitle(r'$k = '+str(kiter)+'$', fontsize=14)
        fig.set_size_inches(12.,8.)
        savefig('B1testmap.png')
        close()
        clf()
        fig, ax = subplots(2,1)
        if newcoords:
            pc1 = ax[0].pcolormesh(xnew, ynew, squeeze(B3new[kz, :, :]), vmin = B3new.min(), vmax = B3new.max())
        else:
            pc1 = ax[0].pcolormesh(xnew, ynew, squeeze(B3[kz, :, :]), vmin = B3.min(), vmax = B3.max())
        colorbar(pc1, ax = ax[0])
        if newcoords:
            pc2 = ax[1].pcolormesh(xnew, ynew, squeeze(B3r[kz, :, :]), vmin = B3new.min(), vmax = B3new.max())
        else:
            pc2 = ax[1].pcolormesh(xnew, ynew, squeeze(B3r[kz, :, :]), vmin = B3.min(), vmax = B3.max())
        colorbar(pc2, ax = ax[1])
        #ax[0].set_xlim(-100., 100.) ; ax[0].set_ylim(-100., 100.)
        # ax[1].set_xlim(-100., 100.) ; ax[1].set_ylim(-100., 100.)
        ax[0].set_xlabel(r'$x$', fontsize=14) ; ax[0].set_ylabel(r'$y$', fontsize=14) 
        ax[1].set_xlabel(r'$x$', fontsize=14) ; ax[1].set_ylabel(r'$y$', fontsize=14) 
        suptitle(r'$k = '+str(kiter)+'$', fontsize=14)
        fig.set_size_inches(16.,10.)
        savefig('B3testmap.png')
        close()
        
    # divergence check:
    divB = (B1r[:,:,1:]-B1r[:,:,:-1])/dx + (B2r[:,1:,:]-B2r[:,:-1,:])/dy + (B3r[1:,:,:]-B3r[:-1,:,:])/dz

    print("maximal expected divergence = ", abs(divB).max())
    
    if interplots:
        clf()
        pcolormesh(xnew, ynew, (divB[:, :, :]).max(axis=0))
        xlabel(r'$x$') ; ylabel(r'$y$') ;  title(r'$\nabla \cdot \bf B$')
        fig.set_size_inches(12.,12.)
        savefig('divBtest.png')
    

sizescale = 128

# Avec(140, ddir = 'D3_lowres', prefix = 'pois', alias=2, interplots=True, omega=1.0, tol=1e-4, tracercutoff = 0., rhocutoff = 0., newcoords=True, xrange=[-280., 280.], yrange=[-280.,280.], zrange=[-280.,280.], nxnew = sizescale*2, nynew = sizescale*2, nznew = sizescale*2, ghostlayers=5, zoomin = 4, enforceraw=True)
# Avec(1000, ddir = 'tintoD3_omega_p6', prefix = 'from_array', alias=2, interplots=True, omega=1.0, tol=1e-3, massboost=1.0, rhocutoff = 1e-7, tracercutoff = 0.1, newcoords = True, xrange=[-1000., 5000.], yrange=[-3000.,1000.], zrange=[-500.,500.], nxnew = sizescale*6, nynew = sizescale*4, nznew = sizescale, ghostlayers=5, zoomin=16)
# Avec(200, ddir = 'tintoD3_omega', prefix = 'from_array', alias=1, omega=1.0, tol=1e-4, massboost=1.0, tracercutoff = 0.9, newcoords = True, xrange=[-1000., 5000.], yrange=[-3000.,1000.], zrange=[-500.,500.], nxnew = sizescale*12, nynew = sizescale*8, nznew = 2*sizescale, ghostlayers=5, zoomin=8, enforceraw = True)
# Avec(7600, ddir = 'tintoD3_high1', prefix = 'from_array', alias=1, omega=1.0, tol=1e-4, massboost=1.0, tracercutoff = 0.9, newcoords = True, xrange=[-2500., 7500.], yrange=[-3000.,1000.], zrange=[-500.,500.], nxnew = sizescale*12, nynew = sizescale*8, nznew = 2*sizescale, ghostlayers=5, zoomin=16, enforceraw = True)
# Avec(600, 120, ddir = 'tintoD3d', prefix = 'from_array', alias=1, interplots=True, omega=1.0, tol=1e-4, rhocutoff = 1e-7, tracercutoff = 0.01, newcoords = True, xrange=[-1000., 1000.], yrange=[-1000.,1000.], zrange=[-200.,200.], nxnew = sizescale*5, nynew = sizescale*5, nznew = sizescale, ghostlayers=5, zoomin=16, massboost=1e3)   
# Avec(600, ddir = 'tintoD3d', prefix = 'from_array', alias=2, interplots=True, omega=1.0, tol=1e-4, massboost=1e3, rhocutoff = 0.0, zoomin=16, ghostlayers = 5) 
# Avec(100, ddir = 'D4', prefix = 'pois', alias=2, interplots=True, omega=1.0, tol=1e-4, massboost=1.0, rhocutoff = 0.0, tracercutoff = 0.0, ghostlayers=5, zoomin=8, newcoords = True, xrange=[-50., 50.], yrange=[-50.,50.], zrange=[-50.,50.], nxnew = sizescale, nynew = 2*sizescale, nznew = sizescale) 
# Avec(100, ddir = 'D4', prefix = 'pois', alias=4, interplots=True, omega=1.0, tol=1e-5, tracercutoff = 0., rhocutoff = 0., newcoords=True, xrange=[-300., 300.], yrange=[-300.,300.], zrange=[-300.,300.], nxnew = 128, nynew = 128, nznew = 128)
# Avec(1300, ddir = 'tintoD3_omega3M', prefix = 'from_array', alias=4, interplots=True, omega=1.0, tol=1e-3, tracercutoff = 0.99, rhocutoff = 1e-7)
