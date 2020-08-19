# -*- coding: utf-8 -*-

'''
******************************************************************************

     =========================================================================
     Subroutine to plot results


     Description                                     Programmer    Date
     ------------------------------------------------------------------
     Original version (1.0).............................NCP  2017/08/XX
     Improved version (2.0)   ..........................NCP  2017/22/11   
    ==========================================================================
    
    The version 2.0 includes:
    - a better description of functions
    - changed the name of some function (except main and craterdimensions)
    - functions:
    
    
    
    ==========================================================================
    
    Need to update this script. We now save the final crater dimensions for
    t/tr = 10 so it's easier to download it

******************************************************************************
'''

# loading of basic Python's module
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import os
from pylab import arange
from matplotlib import rc
import matplotlib.gridspec as gridspec
import copy
import glob


pathm = ['/work/nilscp/Python/prog/clean', '/work/nilscp/iSALE/Dellen/lib']

for pat in pathm:
    sys.path.append(pat)

import pySALEPlot as psp
import crater as cr
import subprocess
'''
***********************************************************************
'''

# Switch on the use of Latex text
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')

'''
***********************************************************************
'''


def make_video(path, delay_number, loop_number, videolabel):
    '''
    description:

    path = '/mypath/' # note the slash at the end
    delay_number: in hundreds of a second
    loop: = 0 infinite looping, 1 = once, 2 = twice ....
    videoname = 'myvideo.gif'  

    example:
    path = '/run/media/nilscp/Squall/benchmarkFI/CPPR10/data/CDILWPO_L100/evolution/plots/Distension/all/'
    delay_number = 10
    loop_number = 1
    videoname = 'distension'
    make_video(path, delay_number, loop_number, videoname)
    '''
    os.chdir(path)

    videoname = videolabel + '_delay' + \
        str(int(delay_number)) + '_loop' + str(loop_number) + '.gif'

    command = ("convert -delay " + str(delay_number) + " -loop " + str(loop_number) + " " + str(path) +
               "*.png " + videoname)

    subprocess.Popen(command.split(), cwd=path)

    print ('Steven Spielberg')


'''
***********************************************************************
'''


def find_nearest(array, value):
    '''
    description:
    find the nearest value in an array

    output:
    value and index where the nearest value is found
    '''

    idx = np.nanargmin((np.abs(array-value)))
    return array[idx], idx


'''
***********************************************************************
'''


def morphology(path1, paths, norm, normpath, timei, showtransient, showhline, lbl):
    '''    
    path1 = '/work/nilscp/iSALE/isaleruns/testruns/ejecta/L20km_part1_HYDRO_CPPR1000/results/L20km_HYDRO_part1g/'
    paths = '/work/nilscp/iSALE/isaleruns/data/ejecta/L20km_HYDRO_part1g/'
    normpath = '/work/nilscp/iSALE/isaleruns/data/ejecta/L20km_HYDRO_part1g/'
    norm = 0.5 # 0.5 by the radius of the projectile?
    showtransient = False
    showhline = False
    timei = 'pen' #'transient' or 'norm' or 'pen' (for penetration time)

    lbl = r"$\mathit{U} = 12.7 km/s, \mathit{L} = 20 km$"

    morphology(path1, paths, norm, normpath, timei, showtransient, showhline, lbl)



    #loop
    L = [100,200,300,400,500,600,700,800,900,1000]
    name = 'C00P10F06LONG_U04_L'

    for l in L:
        tmp_name = name + str(l)
         path1 = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/velocity/results/' + tmp_name
         paths = '/run/media/nilscp/Zell/velocity/data/'  + tmp_name
         lbl = r'$\mathit{U} = 400 m/s, \mathit{L} = ' + str(l) + ' m$"
         evoplot(path1,paths, path1tr, pathstr, norm, extentx, extenty, lbl)


    ############################3

    What if it is nan?

    What if we want to plot the transient crater in addition

    Maybe I could also have a function without having the need to have extentx and extenty
    taking 20% of the final drim and so much of the maximum depth.... or a fraction of the final drim
    we known that d ~ 0.2D

    norm = 0.5 can only be used for showtransient == False
    '''

    plt.ioff()  # figures does not pop up

    os.chdir(path1)
    mod1 = psp.opendatfile('jdata.dat')

    modelname = path1.split('/')[-2]

    if not os.path.exists(paths + 'evolution/plots/morphology'):
        os.makedirs(paths + 'evolution/plots/morphology')

    if (norm == 0):
        if not os.path.exists(paths + 'evolution/plots/morphology/absolute/'):
            os.makedirs(paths + 'evolution/plots/morphology/absolute')
    elif (norm == 0.5):
        if not os.path.exists(paths + 'evolution/plots/morphology/norm_proj/'):
            os.makedirs(paths + 'evolution/plots/morphology/norm_proj')
    elif (norm == 1):
        if not os.path.exists(paths + 'evolution/plots/morphology/norm_its/'):
            os.makedirs(paths + 'evolution/plots/morphology/norm_its')
    else:
        if not os.path.exists(paths + 'evolution/plots/morphology/norm/'):
            os.makedirs(paths + 'evolution/plots/morphology/norm')

    if norm == 1:
        os.chdir(paths + 'transient/')

        # load data for the transient crater 1
        t_trg, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __ = np.loadtxt(
            modelname + '_tr.txt', delimiter='\t', comments='#')

        os.chdir(paths + 'final/')
        t_f, __, __, __, __, drim_f, __ = np.loadtxt(
            modelname + '_final.txt', delimiter='\t', comments='#')
        rinner1 = drim_f/2.

    elif norm == 2:

        os.chdir(paths + 'transient/')

        # load data for the transient crater
        t_trg, __, __, __, __, __, __, __ = np.loadtxt(
            modelname + '_tr.txt', delimiter='\t', comments='#')

        modelnametr = normpath.split('/')[-2]
        os.chdir(normpath + 'transient/')

        # load data for the transient crater modtr
        t_tr1, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __ = np.loadtxt(
            modelnametr + '_tr.txt', delimiter='\t', comments='#')

        os.chdir(normpath + 'final/')
        t_fnorm, __, __, __, __, drim_fnorm, __ = np.loadtxt(
            modelnametr + '_final.txt', delimiter='\t', comments='#')
        rinner1 = drim_fnorm/2.

    else:
        os.chdir(paths + 'transient/')

        # load data for the transient crater 1
        t_trg, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __ = np.loadtxt(
            modelname + '_tr.txt', delimiter='\t', comments='#')

        os.chdir(paths + 'final/')
        t_f, __, __, __, __, drim_f, __ = np.loadtxt(
            modelname + '_final.txt', delimiter='\t', comments='#')

    # if we want to show the transient crater
    if showtransient:
        os.chdir(paths + 'transient/')
        dataXY = np.loadtxt(
            modelname + '_XYtransientprofile.txt', delimiter='\t', comments='#')
        X = dataXY[:, 0]
        Y = dataXY[:, 1]

    os.chdir(path1)
    for i in range(mod1.nsteps):
        step = mod1.readStep('Den', i)

        t1 = step.time
        # plotting of the data
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111)

        if norm >= 1:
            ax1.pcolormesh(mod1.x/rinner1, mod1.y/rinner1, step.mat,
                           cmap='BrBG_r', vmin=-5., vmax=5., zorder=2)
            ax1.contour(mod1.xc/rinner1, mod1.yc/rinner1,
                        step.cmc[0], 1, colors='b', linewidths=4, zorder=4)
            if t1 >= t_trg:
                if showtransient:
                    # I need to show it only for a range
                    ax1.plot(X/rinner1, Y/rinner1, "ro")
                else:
                    None
            #ij1 = np.where((data_tr1[:,1]<=500.) & (data_tr1[:,0]<=Dr_tr1/2.))
            #ij2 = np.where((data_tr2[:,1]<=500.) & (data_tr2[:,0]<=Dr_tr2/2.))

            if showhline:
                ax1.hlines(0, 0, 1.2, 'r', linewidths=4, zorder=1)
            else:
                None

            ax1.set_xlim([0, 1.2])  # -2000,0 ## -1500,1500

            for ax in [ax1]:
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16, length=10,
                               width=2., which='major')
                ax.set_ylim([-0.8, 0.8])

            for u in range(mod1.tracer_numu):
                tru = mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in np.arange(0, len(tru.xlines), 20):
                    ax1.plot(step.xmark[tru.xlines[l]]/rinner1,
                             step.ymark[tru.xlines[l]]/rinner1,
                             c='k', marker='.', linestyle='None', markersize=1.0, zorder=3)

                # Plot the tracers in horizontal lines, every 20 lines
                for l in np.arange(0, len(tru.ylines), 20):
                    ax1.plot(step.xmark[tru.ylines[l]]/rinner1,
                             step.ymark[tru.ylines[l]]/rinner1,
                             c='k', marker='.', linestyle='None', markersize=1.0, zorder=3)

        elif norm == 0.5:
            ax1.pcolormesh(mod1.x/(mod1.cppr[0]*mod1.dx), mod1.y/(
                mod1.cppr[0]*mod1.dx), step.mat, cmap='BrBG_r', vmin=-5., vmax=5., zorder=2)
            ax1.contour(mod1.xc/(mod1.cppr[0]*mod1.dx), mod1.yc/(
                mod1.cppr[0]*mod1.dx), step.cmc[0], 1, colors='b', linewidths=4, zorder=4)
            if t1 >= t_trg:
                if showtransient:
                    # I need to show it only for a range
                    ax1.plot(X/(mod1.cppr[0]*mod1.dx),
                             Y/(mod1.cppr[0]*mod1.dx), "ro")
                else:
                    None

            if showhline:
                ax1.hlines(
                    0, 0, mod1.xhires[1]/(mod1.cppr[0]*mod1.dx), 'r', linewidths=4, zorder=1)
            else:
                None
            # -2000,0 ## -1500,1500
            ax1.set_xlim([0, mod1.xhires[1]/(mod1.cppr[0]*mod1.dx)])

            for ax in [ax1]:
                ax.set_ylim([mod1.yhires[0]/(mod1.cppr[0]*mod1.dy),
                             mod1.yhires[1]/(mod1.cppr[0]*mod1.dy)])
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16, length=10,
                               width=2., which='major')

            for u in range(mod1.tracer_numu):
                tru = mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in np.arange(0, len(tru.xlines), 20):
                    ax1.plot(step.xmark[tru.xlines[l]]/(mod1.cppr[0]*mod1.dx),
                             step.ymark[tru.xlines[l]]/(mod1.cppr[0]*mod1.dx),
                             c='k', marker='.', linestyle='None', markersize=1.0, zorder=3)

                # Plot the tracers in horizontal lines, every 20 lines
                for l in np.arange(0, len(tru.ylines), 20):
                    ax1.plot(step.xmark[tru.ylines[l]]/(mod1.cppr[0]*mod1.dx),
                             step.ymark[tru.ylines[l]]/(mod1.cppr[0]*mod1.dx),
                             c='k', marker='.', linestyle='None', markersize=1.0, zorder=3)

        else:
            ax1.pcolormesh(mod1.x, mod1.y, step.mat,
                           cmap='BrBG_r', vmin=-5., vmax=5., zorder=2)
            ax1.contour(mod1.xc, mod1.yc,
                        step.cmc[0], 1, colors='b', linewidths=4, zorder=4)
            if t1 >= t_trg:
                if showtransient:
                    ax1.plot(X, Y, "ro")  # I need to show it only for a range
                else:
                    None

            if showhline:
                ax1.hlines(0, 0, (drim_f/2.)*1.2, 'r', linewidths=4, zorder=1)
            else:
                None
            ax1.set_xlim([0, (drim_f/2.)*1.2])  # -2000,0 ## -1500,1500

            for ax in [ax1]:
                ax.set_ylim([-(drim_f/2.)*0.8, (drim_f/2.)*0.8])
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16, length=10,
                               width=2., which='major')

            for u in range(mod1.tracer_numu):
                tru = mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in np.arange(0, len(tru.xlines), 20):
                    ax1.plot(step.xmark[tru.xlines[l]],
                             step.ymark[tru.xlines[l]],
                             c='k', marker='.', linestyle='None', markersize=1.0, zorder=3)

                # Plot the tracers in horizontal lines, every 20 lines
                for l in np.arange(0, len(tru.ylines), 20):
                    ax1.plot(step.xmark[tru.ylines[l]],
                             step.ymark[tru.ylines[l]],
                             c='k', marker='.', linestyle='None', markersize=1.0, zorder=3)

        if norm == 0:
            ax1.set_xlabel("x (m)", fontsize=20)
            ax1.set_ylabel("y (m)", fontsize=20)
            fig.tight_layout()

        elif norm == 0.5:
            ax1.set_xlabel(r"$x / a$", fontsize=20)
            ax1.set_ylabel(r"$y / a$", fontsize=20)
            fig.tight_layout()

        else:
            ax1.set_xlabel(r"$x / R_{r}$", fontsize=20)
            ax1.set_ylabel(r"$y / R_{r}$", fontsize=20)
            fig.tight_layout()

        if timei == 'normal':
            ax1.set_title(r"$\mathit{t} = " + str(np.around(t1, decimals=2)
                                                  ) + " " + " s$", position=(0.5, 0.9), fontsize=18)
        elif timei == 'pen':
            ax1.set_title(r"$\mathit{t} = " + str(np.around(t1, decimals=2)) + " " + " s, $" + " " + "$t / t_s = $" + str(
                np.around(t1/((mod1.cppr[0]*mod1.dx*2.) / (mod1.objvel)), decimals=2)), position=(0.5, 0.9), fontsize=18)
        else:
            ax1.set_title(r"$\mathit{t} = " + str(np.around(t1, decimals=2)) + " " + " s, $" + " " +
                          "$\zeta$ = $" + str(np.around(t1/t_trg, decimals=2)), position=(0.5, 0.9), fontsize=18)

        st = fig.suptitle(lbl, fontsize=22)
        st.set_y(0.97)
        st.set_x(0.55)
        fig.subplots_adjust(top=0.9)

        if norm == 0:
            figtitle = modelname + '_' + str(int(i)).zfill(3) + ".png"
            fig.savefig(
                paths + 'evolution/plots/morphology/absolute/' + figtitle, dpi=300)
        elif norm == 0.5:
            figtitle = modelname + '_' + str(int(i)).zfill(3) + ".png"
            fig.savefig(
                paths + 'evolution/plots/morphology/norm_proj/' + figtitle, dpi=300)
        elif norm == 1:
            figtitle = 'itself_norm_' + modelname + \
                '_' + str(int(i)).zfill(3) + ".png"
            fig.savefig(
                paths + 'evolution/plots/morphology/norm_its/' + figtitle, dpi=300)
        else:
            figtitle = 'norm_' + modelname + \
                '_' + str(int(i)).zfill(3) + ".png"
            fig.savefig(
                paths + 'evolution/plots/morphology/norm/' + figtitle, dpi=300)
        plt.close()

    mod1.closeFile()


'''
***********************************************************************
'''


def zoom(zoom_id, norm, mod, rrim, manualx, manualy):
    '''
    example:
    zoom_id = 'hires' # 'close'; 'mid', 'hires'; 'all'
    norm = 2
    rrim = rinner1
    manualx = ((0,1500.)) # if norm not equal to 0 then it has to be in normalized distance!
    manualy = ((-1000., 500.)) # if norm not equal to 0 then it has to be in normalized distance!
    extentx, extenty = zoom(zoom_id, norm, mod1, rrim, manualx, manualy)
    
    added manual opportunity?
    
    if manual_entry
    '''
    
    # create empty arrays
    extentx = []
    extenty = []

    zid = np.array(['close', 'mid', 'hires', 'all', 'manual'])
    
    
    for z in zoom_id:

        idx = np.where(zid == z)[0][0]
    
        if norm == 0:
            if idx == 0:
                extentx.append((0.0, 1.2*rrim))
                extenty.append((-0.8*rrim, 0.8*rrim))
    
            elif idx == 1:
                extentx.append((0.0, 2.0*rrim))
                extenty.append((-2.0*rrim, 0.8*rrim))
    
            elif idx == 2:
                extentx.append((0.0, mod.xhires[1]))
                extenty.append((mod.yhires[0], mod.yhires[1]))
                
            elif idx == 4:
                extentx.append(manualx)
                extenty.append(manualy)
    
            else:
                extentx.append((0.0, np.max(mod.xc)))
                extenty.append((np.min(mod.yc), np.max(mod.yc)))
    
        else:
            if idx == 0:
                extentx.append((0.0, 1.2))
                extenty.append((-0.8, 0.8))
    
            elif idx == 1:
                extentx.append((0.0, 2.0))
                extenty.append((-2.0, 0.8))
    
            elif idx == 2:
                extentx.append((0.0, mod.xhires[1]/rrim))
                extenty.append((mod.yhires[0]/rrim, mod.yhires[1]/rrim))
                
            elif idx == 4:
                extentx.append(manualx)
                extenty.append(manualy)
    
            else:
                extentx.append((0.0, np.max(mod.xc)/rrim))
                extenty.append((np.min(mod.yc)/rrim, np.max(mod.yc)/rrim))
        
    ix_to_delete = []        
    for iy, y in enumerate(extenty):
        if np.isnan(extenty[iy][1]):
            ix_to_delete.append(iy)
                
    if len(ix_to_delete) > 0:
        zoom_id_up = np.delete(zoom_id, ix_to_delete)
            
        bb = np.array(range(len(zoom_id)))
        aa = np.array(ix_to_delete)
        cc = np.setdiff1d(bb,aa)
            
        extentxup = []
        extentyup = []
            
        for c in cc:
            extentxup.append(extentx[c])
            extentyup.append(extenty[c])
                
    else:
        extentxup = extentx
        extentyup = extenty
        zoom_id_up = zoom_id
            
    return extentxup, extentyup, zoom_id_up


'''
***********************************************************************
'''


def field_definition(model, zoom_id, paths, param = 'all'):
    '''
    example:
    param = 'Por'
    paths = '/run/media/nilscp/Squall/benchmarkFI/CPPR10/data/CDILWPO_L100/'
    zoom_id = ['mid', 'hires']

    fld_param, fld_name, fld_cmap, fld_unit, fld_factor, npath = field_definition(model, zoom_id, paths, param = 'all')
    
    #Updated field_definition the 24th June 2019
    
    '''

    fld = np.array(['Cm1', 'Cm2', 'Den', 'Tmp', 'Pre', 'Sie', 'Yld',
                    'Dam', 'Alp', 'TPS', 'YAc'])

    fld_param = np.array(['Cm1', 'Cm2', 'Den', 'Tmp', 'Pre', 'Sie', 'Yld',
                          'Dam', 'Alp', 'TPS', 'YAc'])

    fld_name = np.array(['material 1', 'material 2', 'density', 'temperature', 'pressure', 'sie',
                         'YS', 'damage', 'distension', 'TPS',
                         'YAc'])

    fld_folder = np.array(['Cm1', 'Cm2', 'Density', 'Temperature', 'Pressure', 'Specific_internal_energy',
                           'Yield_strength', 'Damage', 'Distension', 'Total_plastic_strain',
                           'Acoustic_fluidisation_strength'])

    fld_cmap = np.array(['BrBG_r', 'BrBG_r', 'viridis', 'autumn', 'bwr', 'bwr',
                         'summer', 'winter', 'gray_r', 'Reds_r',
                         'summer'])

    fld_unit = np.array(['Conc. material','Conc. material','kg/m^{3}', 'Kelvin', 'GPa', 'MJ',
                         'GPa', 'Damage', 'Distention', 'Strain',
                         'MPa'])

    fld_factor = np.array([1., 1., 1., 1., 1e9, 1e6, 1e9, 1., 1., 1., 1e6])
    
    
    # In case for all the parameters 
    if param == 'all':
        
        # get all the field available for the model
        iSALEparam0 = model.fieldlist
        iSALEparam = []
    
        for param in iSALEparam0:
            if ((param[0].startswith('V_')) | (param[0].startswith('Tr'))):
                None
            else:
                iSALEparam.append(param[0])
                
        # get the lists
        param1 = []
        name = []
        cmap = []
        unit = []
        factor = []
        npathl = []
        
        for parami in iSALEparam:
            
            idx = np.where(fld == parami)[0][0]
        
            param1.append(fld_param[idx])
            name.append(fld_name[idx])
            cmap.append(fld_cmap[idx])
            unit.append(fld_unit[idx])
            factor.append(fld_factor[idx])
            
            
            for z in zoom_id:
                npath = paths + 'evolution/plots/' + fld_folder[idx] + '/' + z + '/'
                npathl.append(npath)
                
                if not os.path.exists(npath):
                    os.makedirs(npath)
                    
    
    else:
        idx = np.where(fld == param)[0][0]
    
        param1 = fld_param[idx]
        name = fld_name[idx]
        cmap = fld_cmap[idx]
        unit = fld_unit[idx]
        factor = fld_factor[idx]
        npathl = []
        
        for z in zoom_id:
            npath = paths + 'evolution/plots/' + fld_folder[idx] + '/' + z + '/'
            npathl.append(npath)
            
            if not os.path.exists(npath):
                os.makedirs(npath)

    return param1, name, cmap, unit, factor, npathl


'''
***********************************************************************
'''

def truncate(n, decimals=0):
    
    '''
    see website: https://realpython.com/python-rounding/
    
    >>> truncate(-5.963, 1)
    -5.9
    
    
    >>> truncate(-1374.25, -3)
    -1000.0
    '''
    
    multiplier = 10 ** float(decimals)
    return int(n * multiplier) / multiplier

'''
***********************************************************************
'''

def vmin_vmax(path1, mod1, fld_param, fld_factor, param):
    
    '''
    Problem with the total plastic strain and the acoustic fluidization strength
    
    '''
    
    
    # change to directory
    os.chdir(path1)
    
    # empty list
    vminlst = []
    vmaxlst = []
        
    # read step only one time
    step = mod1.readStep(fld_param, 1)
    
    
    for ifld, fld in enumerate(fld_param):
        
        if fld.startswith('Cm'):       
            vminlst.append(0.0)
            vmaxlst.append(1.0)
            
        elif fld.startswith('Al'):
            vminlst.append(1.00)
            vmaxlst.append(1.25)
            
        elif fld.startswith('Da'):
            vminlst.append(0.0)
            vmaxlst.append(1.0)
            
        elif fld.startswith('TP'):
            vminlst.append(1e-5)
            vmaxlst.append(1.0)
            
        elif fld.startswith('Pr'):
            vminlst.append(0.0)
            vmaxlst.append(1.0)
            
        elif fld.startswith('Yl'):
            vminlst.append(0.0)
            vmaxlst.append(1.0) 
        else:
            
            # could I automatically find vmiin and vmaax
            ixfld = np.where(step.cmc[0] >= 0.5) # at least contain 50% of material
            
            # automatically find vmin and vmax
            vminlst.append(0.0)
            vmaax = np.percentile(step.data[ifld][ixfld]/fld_factor[ifld], 70)
            
            
            # rounding 
            if vmaax >= 1000.:
                vmaaxg = truncate(vmaax, -2)
            elif ((vmaax >= 10.) & (vmaax < 1000.)):
                vmaaxg = truncate(vmaax, -1)
            elif ((vmaax >= 0.1) & (vmaax < 10.)):
                vmaaxg = truncate(vmaax, 1)
            else:
                vmaaxg = truncate(vmaax, 3)
            
            # if somehow the rounding does not work then just use 1
            if vmaaxg == 0.0:
                vmaxlst.append(1.0)
            else:
                vmaxlst.append(vmaaxg)
            
    return (vminlst, vmaxlst)
    
'''
***********************************************************************
'''


def field(path1, paths, norm, normpath, zoom_id, lbl, vmiin, vmaax, manualx, manualy, param = 'all'):
    '''
    param = 'Den'  
    path1 = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/benchmarkFI/CPPR/UAVG/CPPR10/CDILWPO/CDILWPO_L100_dev_larger_cells/'
    paths = '/run/media/nilscp/Squall/benchmarkFI/CPPR10/data/CDILWPO_L100_dev_larger_cells/'
    normpath = '/run/media/nilscp/Squall/benchmarkFI/CPPR10/data/CDILWPO_L100/'
    norm = 0
    zoom_id = ['close', 'mid', 'hires', 'all', 'manual'] # 'mid'; 'hires'; 'all' #depending on the zoom
    param = 'all'
    vmiin = [] # or vmiin = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1e-5, 0.0] # need to take into account the field factor
    vmaax = [] # or vmaax = [1.0, 1.0, 2000.0, 400.0, 1.0, 1.0, 1.0, 1.0, 1.25,1.0, 1.0]

    lbl = r"$\mathit{L} = 100 m"

    field(path1, paths, norm, normpath, zoom_id, lbl, vmiin, vmaax, param)
    
    I should replace vmiin and vmaax by percentile values that are rounded to the hundreds


    Field list:
    [['Cm1', 'Concentration, mat. 1'],
     ['Cm2', 'Concentration, mat. 2'],
     ['Den', 'Density'],
     ['Tmp', 'Temperature'],
     ['Pre', 'Pressure'],
     ['Sie', 'Specific internal energy'],
     ['Yld', 'Yield strength'],
     ['Dam', 'Damage'],
     ['Alp', 'Distension'],
     ['TPS', 'Total plastic strain (shear)'],
     ['YAc', 'Acoustic fluidisation strength'],
     ['V_x', 'Velocity (x)'],
     ['V_y', 'Velocity (y)'],
     ['Trx', 'Tracer x position'],
     ['Try', 'Tracer y position'],
     ['TrP', 'Tracer peak pressure'],
     ['TrT', 'Tracer peak temperature']]
    '''

    plt.ioff()  # figures does not pop up

    os.chdir(path1)
    mod1 = psp.opendatfile('jdata.dat')

    # modelname
    modelname = path1.split('/')[-2]

    # loading ot the data
    ###########################################################################
    if norm == 1:
        os.chdir(paths + 'transient/')

        # load data for the transient crater 1
        t_trg, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __ = np.loadtxt(
            modelname + '_tr.txt', delimiter='\t', comments='#')

        os.chdir(paths + 'final/')
        t_f, __, __, __, __, drim_f, __ = np.loadtxt(
            modelname + '_final040.txt', delimiter='\t', comments='#') # put to 040 here
        
        rinner1 = drim_f/2.

    elif norm == 2:

        os.chdir(paths + 'transient/')

        # load data for the transient crater
        t_trg, __, __, __, __, __, __, __ = np.loadtxt(
            modelname + '_tr.txt', delimiter='\t', comments='#')

        modelnametr = normpath.split('/')[-2]
        os.chdir(normpath + 'transient/')

        # load data for the transient crater modtr
        t_tr1, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __ = np.loadtxt(
            modelnametr + '_tr.txt', delimiter='\t', comments='#')
        
        # is it normal that it only load the data for the norm path?
        # maybe I guess so
        os.chdir(normpath + 'final/')
        t_fnorm, __, __, __, __, drim_fnorm, __ = np.loadtxt(
            modelnametr + '_final040.txt', delimiter='\t', comments='#') # put to 040 here
        
        rinner1 = drim_fnorm/2.

    else:
        os.chdir(paths + 'transient/')

        # load data for the transient crater 1
        t_trg, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __ = np.loadtxt(
            modelname + '_tr.txt', delimiter='\t', comments='#')

        os.chdir(paths + 'final/')
        t_f, __, __, __, __, drim_f, __ = np.loadtxt(
            modelname + '_final040.txt', delimiter='\t', comments='#') # put to 040 here
            
        rinner1 = drim_f/2.

    ###########################################################################
    
    try:
        # choosing the degree of the zooming on the figure
        # get the extent
        extentx, extenty, zoom_id_up = zoom(zoom_id, norm, mod1, rinner1, manualx, manualy)
        
        # field list (should I make a function out of it?)
        fld_param, fld_name, fld_cmap, fld_unit, fld_factor, npath = field_definition(mod1, zoom_id_up, paths, param)

        # update label
        nlbl = []
        
        # small change (for case with only one selected parameter)
        if param != 'all':
            nlbl.append(lbl + ", " + fld_name + "$")
            fld_param = [fld_param]
            fld_factor = [fld_factor]
            fld_name = [fld_name]
            fld_cmap = [fld_cmap]
            fld_unit = [fld_unit]
        else:        
            for fname in fld_name:
                nlbl.append(lbl + ", " + fname + "$")
        
        # print fld_param
        print (fld_param)


        

        
        #vmin and vmax (update here)
        if len(vmiin) == 0:
            vmiin, vmaax = vmin_vmax(path1, mod1, fld_param, fld_factor, param)
        else:
            None
    
        # looping of
        os.chdir(path1)
        for i in range(mod1.nsteps):
            step = mod1.readStep(fld_param, i)
    
            t1 = step.time
            # plotting of the data
    
            # changing the vmin and vmax every step will change the value everytime
            # maybe I should define manually
            
            if norm >= 1:
                
                '''
                After the change including all parameters this statement will never
                be fullfill expect when param = 'Por'
                '''
                
                # don't like the 'Por' thing
                #if param == 'Por':
                #    cax = ax1.pcolormesh(mod1.x/rinner1, mod1.y/rinner1, (1. - (1./step.data[0]))*100.,
                #                         vmin=vmiin,
                #                         vmax=vmaax,
                #                         cmap=fld_cmap, zorder=2)
                
                count = 0
                
                for ifld, fld in enumerate(fld_param):
                    
                    for iz, z in enumerate(zoom_id_up):
                        
                        fig = plt.figure(figsize=(6, 6))
                        ax1 = fig.add_subplot(111)
                        
                        # plot data
                        cax = ax1.pcolormesh(mod1.x/rinner1, mod1.y/rinner1, step.data[ifld]/fld_factor[ifld],
                                                 vmin=vmiin[ifld],
                                                 vmax=vmaax[ifld],
                                                 cmap=fld_cmap[ifld], zorder=2)
        
                        ax1.contour(mod1.xc/rinner1, mod1.yc/rinner1,
                                step.cmc[0], 1, colors='k', linewidths=2, zorder=4)
                    
                    
        
                        for ax in [ax1]:
                            ax.minorticks_off()
                            ax.tick_params('both', labelsize=16, length=10,
                                           width=2., which='major')
                            
                        cbar = fig.colorbar(cax)
                        cbar.ax.tick_params(labelsize=15)
                        cbar.set_label('$' + fld_unit[ifld] + '$', labelpad=-
                                       42.5, y=1.07, fontsize=15, rotation=0)
                        
                        # set labels
                        ax1.set_xlabel(r"$x / R_{r}$", fontsize=20)
                        ax1.set_ylabel(r"$y / R_{r}$", fontsize=20)
                        
                        # set title            
                        ax1.set_title(r"$\mathit{t} = " + "{:.2f}".format(t1) + " " + " s, $" + " " +
                                      "$\zeta$ = " + "{:.2f}".format(t1/t_trg), position=(0.5, 0.9), fontsize=18)
                        
                        # related to the zoom
                        ax1.set_xlim(extentx[iz])
                        for ax in [ax1]:
                            ax.set_ylim(extenty[iz])
                        fig.tight_layout()
                        
                        st = fig.suptitle(nlbl[ifld], fontsize=20)
                        st.set_y(0.97)
                        st.set_x(0.5)
                        fig.subplots_adjust(top=0.9)
                            
                                                    
                        if norm == 1:
                            figtitle = 'itself_norm_' + modelname + \
                                '_' + str(int(i)).zfill(3) + ".png"
                            fig.savefig(npath[count] + figtitle, dpi=300)
                            
                        elif norm == 2:
                            figtitle = 'norm_' + modelname + \
                                '_' + str(int(i)).zfill(3) + ".png"
                            fig.savefig(npath[count] + figtitle, dpi=300)
                            
                        plt.close()
                        
                        count = count + 1
    
    
            else:
                count = 0
                
                for ifld, fld in enumerate(fld_param):
                    
                    for iz, z in enumerate(zoom_id_up):
                        
                        fig = plt.figure(figsize=(6, 6))
                        ax1 = fig.add_subplot(111)
                        
                        # plot data
                        cax = ax1.pcolormesh(mod1.x, mod1.y, step.data[ifld]/fld_factor[ifld],
                                                 vmin=vmiin[ifld],
                                                 vmax=vmaax[ifld],
                                                 cmap=fld_cmap[ifld], zorder=2)
        
                        ax1.contour(mod1.xc, mod1.yc,
                                step.cmc[0], 1, colors='k', linewidths=2, zorder=4)
                    
                    
        
                        for ax in [ax1]:
                            ax.minorticks_off()
                            ax.tick_params('both', labelsize=16, length=10,
                                           width=2., which='major')
                            
                        cbar = fig.colorbar(cax)
                        cbar.ax.tick_params(labelsize=15)
                        cbar.set_label('$' + fld_unit[ifld] + '$', labelpad=-
                                       42.5, y=1.07, fontsize=15, rotation=0)
                        
                        # set labels
                        ax1.set_xlabel(r"$x$", fontsize=20)
                        ax1.set_ylabel(r"$y$", fontsize=20)
                        
                        # set title            
                        ax1.set_title(r"$\mathit{t} = " + "{:.2f}".format(t1) + " " + " s$", position=(0.5, 0.9), fontsize=18)
                        
                        # related to the zoom
                        ax1.set_xlim(extentx[iz])
                        for ax in [ax1]:
                            ax.set_ylim(extenty[iz])
                        fig.tight_layout()
                        
                        st = fig.suptitle(nlbl[ifld], fontsize=20)
                        st.set_y(0.97)
                        st.set_x(0.5)
                        fig.subplots_adjust(top=0.9)
                            
                        figtitle = modelname + '_' + str(int(i)).zfill(3) + ".png"
                        fig.savefig(npath[count] + figtitle, dpi=300)
                        
                            
                        plt.close()
                        
                        count = count + 1
    except:
        None

    mod1.closeFile()


'''
***********************************************************************
'''


def mainevo(folders, path1, paths, norm, normpath, zoom_id, param):
    
    '''
    path1 = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/benchmarkFI/CPPR/UAVG/CPPR10/'
    paths = '/run/media/nilscp/Squall/benchmarkFI/CPPR10/data/'
    normpath = ''
    norm = 0
    zoom_id = ['close', 'all', 'mid', 'hires'] # 'mid'; 'hires'; 'all' #depending on the zoom
    param = 'all'
    lbl = r"$\mathit{L} = 100 m"
    L = [100]
    
    # maybe detect automatically from what stands behind L 
    
    Need to be tested
    
    
    '''
    os.chdir(path1)
    
    folders = glob.glob('*')
    
    for f in folders:
        os.chdir(path1 + '/' + f)
        modelnames = glob.glob('*')
        print (modelnames)
     
        for im, modelname in enumerate(modelnames):
            
            pathsa = paths + modelname + '/'
            path1a = path1 + f + '/' + modelname + '/'
            
            L = modelname.split('_L')[1]
            Ln = L.split('_')[0]
            
            #normpath should be the same
            lbla = r"$\mathit{L} = " + Ln + " m"
                        
            field(path1a, pathsa, norm, normpath, zoom_id, lbla, param = 'all')
        
        

def main(pathdata, folders, pathplots):
    '''
    description:
    routine that plots both the transient, excavated and final cavities.
    This routine is useful to see whether the different crater dimensions by
    the scripts in crater.py and ejecta.py managed to perform well (i.e., this is
    visual quality assesment of the results)

    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)

    folders: all modelcases that need to be plotted

    pathplots: saving plot directory
    '''

    for modelname in folders:
        transient(pathdata, modelname, pathplots)
        final(pathdata, modelname, pathplots)
        excavated(pathdata, modelname, pathplots)

    print ("ROCK'N ROLL")


'''
***********************************************************************
'''


def transient(pathdata, modelname, pathplots):
    '''
    description:
    routine that plots transient crater dimensions

    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)

    folders: all modelcases that need to be plotted

    pathplots: saving plot directory

    example:
    pathdata = '/media/nilscp/Zell/Collapse/data/'
    modelname = 'C00P20F08_L250'
    pathplots = '/media/nilscp/Zell/Collapse/rplots/'
    transient(pathdata,modelname,pathplots)
    '''

    # should be outside of the loop
    if not os.path.exists(pathplots):
        os.makedirs(pathplots)

    os.chdir(pathdata + modelname + '/transient/')

    t, da, Da, V, h, Dr, Vr, __ = np.loadtxt(
        modelname + '_tr.txt', delimiter='\t', comments='#')
    dataXY = np.loadtxt(modelname + '_XYtransientprofile.txt',
                        delimiter='\t', comments='#')
    X = dataXY[:, 0]
    Y = dataXY[:, 1]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y, "ro", zorder=2)
    ax.hlines(0, np.min(X), np.max(X), 'k', linewidth=3)
    ax.plot(Da/2., 0, 'yo', ms=10, label='Ra: apparent radius')
    ax.plot(Dr/2., h, 'co', ms=10, label='Drim and hrim')
    ax.hlines(-da, np.min(X), np.max(X), 'b',
              linewidth=3, label='da: apparent depth')
    ax.legend(loc='center right')
    ax.set_xlabel('X (m)', fontsize=18)
    ax.set_ylabel('Y (m)', fontsize=18)
    ax.tick_params('both', labelsize=16, length=10, width=1., which='major')
    fig.tight_layout()
    st = fig.suptitle(modelname + ' transient profile', fontsize=18)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    figtitle = modelname + ' transient_profile.png'
    fig.savefig(pathplots+figtitle, dpi=300)
    plt.close()


'''
***********************************************************************
'''


def final(pathdata, modelname, pathplots):
    '''
    description:
    routine that plots final crater dimensions

    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)

    folders: all modelcases that need to be plotted

    pathplots: saving plot directory

    example:
    pathdata = '/media/nilscp/Zell/Collapse/data/'
    modelname = 'C00P20F08_L250'
    pathplots = '/media/nilscp/Zell/Collapse/rplots/'
    final(pathdata,modelname,pathplots)
    '''

    # should be outside of the loop
    if not os.path.exists(pathplots):
        os.makedirs(pathplots)

    os.chdir(pathdata + modelname + '/final/')

    t, da, Da, V, h, Dr, Vr = np.loadtxt(
        modelname + '_final.txt', delimiter='\t', comments='#')
    dataXY = np.loadtxt(modelname + '_XYfinalprofile.txt',
                        delimiter='\t', comments='#')
    X = dataXY[:, 0]
    Y = dataXY[:, 1]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y, "ro", zorder=2)
    ax.hlines(0, np.min(X), np.max(X), 'k', linewidth=3)
    ax.plot(Da/2., 0, 'yo', ms=10, label='Ra: apparent radius')
    ax.plot(Dr/2., h, 'co', ms=10, label='Drim and hrim')
    ax.hlines(-da, np.min(X), np.max(X), 'b',
              linewidth=3, label='da: apparent depth')
    ax.legend(loc='center right')
    ax.set_xlabel('X (m)', fontsize=18)
    ax.set_ylabel('Y (m)', fontsize=18)
    ax.tick_params('both', labelsize=16, length=10, width=1., which='major')
    fig.tight_layout()
    st = fig.suptitle(modelname + ' final profile', fontsize=18)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    figtitle = modelname + ' final_profile.png'
    fig.savefig(pathplots+figtitle, dpi=300)
    plt.close()


'''
***********************************************************************
'''


def excavated(pathdata, modelname, pathplots):
    '''
    description:
    routine that plots the excavated crater dimensions

    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)

    folders: all modelcases that need to be plotted

    pathplots: saving plot directory

    example:
    pathdata = '/media/nilscp/Zell/Collapse/data/'
    modelname = 'C00P20F08_L250'
    pathplots = '/media/nilscp/Zell/Collapse/rplots/'
    excavated(pathdata,modelname,pathplots)
    '''

    # should be outside of the loop
    if not os.path.exists(pathplots):
        os.makedirs(pathplots)

    os.chdir(pathdata + modelname + '/excavated/')

    # load excavated data
    # possible miss something at the end
    Ve, de, De = np.loadtxt(modelname + '_excavated.txt',
                            delimiter='\t', comments="#")
    Z = np.loadtxt(modelname + '_tracersXY_excavated.txt', delimiter='\t',
                   comments="#")  # possible miss something at the end
    X = Z[:, 0]
    Y = Z[:, 1]

    # load transient profile
    os.chdir(pathdata + modelname + '/transient/')
    dataXY = np.loadtxt(modelname + '_XYtransientprofile.txt',
                        delimiter='\t', comments='#')
    X2 = dataXY[:, 0]
    Y2 = dataXY[:, 1]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(X, Y, "bo")
    ax.plot(X2, Y2, "ro")
    ax.hlines(de, np.min(X2), np.max(X2), 'b', linewidth=3,
              label='de: apparent excavated depth')
    ax.plot(De/2., 0, 'yo', ms=10, label='Re: apparent excavated radius')
    ax.legend(loc='lower right')
    ax.set_xlabel('X (m)', fontsize=18)
    ax.set_ylabel('Y (m)', fontsize=18)
    ax.tick_params('both', labelsize=16, length=10, width=1., which='major')
    fig.tight_layout()
    st = fig.suptitle(
        modelname + ' excavated and transient profile', fontsize=18)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    figtitle = modelname + ' excavated_profile.png'
    fig.savefig(pathplots+figtitle, dpi=300)
    plt.close()


'''
***********************************************************************

def comparisonTransientFinal(path1,path2,pathdata,pathtr,paths,figname,
                             modelname1,modelname2,extentx,extenty,black,norm,transient):
    
    ###
    description:
    routine that plots both the transient and final crater dimensions for 
    two different modelcases (modelcase 1 is shown in the left panel and 
    modelcase 2 is shown in the right panel).
    
    This function is pratical if for example the same 
    
    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)
    
    folders: all modelcases that need to be plotted
    
    pathplots: saving plot directory
    
    example:
    
    path1 = '/var/tmp/mysshfs/stallo/layering/AUG/collapse/results/C00P10F06LONG_L500/'
    path2 = '/var/tmp/mysshfs/stallo/layering/AUG/collapse/results/C00P10F07LONG_L500/'
    pathdata = '/media/nilscp/Zell/Collapse/data/'
    pathtr = pathdata #or
    #pathtr = '/media/nilscp/Zell/Collapse/data_transient/'
    paths="/work/nilscp/tmp/"
    
    figname = ''
    modelname1= r'f = 0.6, L = 500 m'
    modelname2= r'f = 0.7, L = 500 m'
    extentx=[0,4000]
    extenty=[-3000,1000]
    norm = '/media/nilscp/Zell/Collapse/data/C00P10F03_L500/final/C00P10F03_L500_final.txt'
    #norm = 2
    transient = True
    
    black = 1 # 1 (black) or 0 (white)
    norm = 0 # 0 (no normalization), normalization to a specific case (string to final .txt file),
    # any other numbers (normalization to the rim-to-rim crater diameter )
    
    comparisonTransientFinal(path1,path2,pathdata,pathtr,paths,figname,
                             modelname1,modelname2,extentx,extenty,black,norm,transient)
    
    # Figure
    black = 1 # figure in black
    black = 0 # figure in white
    
    # norm
    norm = 0 nothing (normal in meters)
    norm = 1 to Drim itself
    norm = string to a specific case

    
    # if you want to plot the transient crater or not
    transient = True
    transient = False 
        
    ###
    # check if path1 and path2 are similar, if yes we only load one of the path
    if path1 == path2:
        os.chdir(path1)
        mod1 = psp.opendatfile('jdata.dat')
        mod2 = psp.opendatfile('jdata.dat')
        step1_1 =  mod1.readStep(['Den'],1)
        step1_2 = mod2.readStep(['Den'],1)
        tstep1 = step1_1.time
        tstep2 = step1_2.time
        lstep1 =  mod1.readStep(['Den'],mod1.nsteps-1)
        lstep2 = mod1.readStep(['Den'],mod2.nsteps-1)
        t1 =  np.around(tstep1*(mod1.nsteps-1),decimals=1)
        t2 = np.around(tstep1*(mod1.nsteps-1),decimals=1)
    else:
           
        # load data for modelcase 1
        os.chdir(path1)   
        mod1=psp.opendatfile('jdata.dat')   
        
        # load data for modelcase 2
        os.chdir(path2)   
        mod2=psp.opendatfile('jdata.dat')
    
        # load first step to get the time
        step1_1 = mod1.readStep(['Den'],1)   
        tstep1 = step1_1.time
        
        step1_2 = mod2.readStep(['Den'],1)
        tstep2 = step1_2.time
        
        #load last step
        lstep1 = mod1.readStep(['Den'],mod1.nsteps-1)
        lstep2 = mod2.readStep(['Den'],mod2.nsteps-1)
        
        # get time last step
        t1 = np.around(tstep1*(mod1.nsteps-1),decimals=1)
        t2 = np.around(tstep2*(mod2.nsteps-1),decimals=1)
    
    # plotting of the data
    if black == 1:
        fig=plt.figure(figsize=(12,6),facecolor='black')
    else:
        fig=plt.figure(figsize=(12,6))
    
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    
    if isinstance(norm, basestring):
                
        #get final rim-to-rim diameter 1
        __, __, __, __, __, Drf1, __  = np.loadtxt(norm,delimiter='\t',comments='#')
        rinner1 = Drf1 /2.
        
        ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,lstep1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
        ax2.pcolormesh(mod2.x/rinner1,mod2.y/rinner1,lstep2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
        
        ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
        ax2.set_xlim([0,1.2])
    
        for ax in [ax1,ax2]:
            ax.minorticks_off()
            ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
            ax.set_ylim([-0.8,0.8])
            
        for u in range(mod1.tracer_numu):
            tru=mod1.tru[u]
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.xlines),20):
                ax1.plot(-lstep1.xmark[tru.xlines[l]]/rinner1,
                        lstep1.ymark[tru.xlines[l]]/rinner1,
                        c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.ylines),20):
                ax1.plot(-lstep1.xmark[tru.ylines[l]]/rinner1,
                         lstep1.ymark[tru.ylines[l]]/rinner1,
                         c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                         
        for u in range(mod2.tracer_numu):
            tru=mod2.tru[u]
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.xlines),20):
                ax2.plot(lstep2.xmark[tru.xlines[l]]/rinner1,
                        lstep2.ymark[tru.xlines[l]]/rinner1,
                        c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.ylines),20):
                ax2.plot(lstep2.xmark[tru.ylines[l]]/rinner1,
                         lstep2.ymark[tru.ylines[l]]/rinner1,
                         c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
    else:        
        if norm == 0:
            ax1.pcolormesh(-mod1.x,mod1.y[::-1],lstep1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax2.pcolormesh(mod2.x,mod2.y,lstep2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            
            ax1.set_xlim([-extentx[1],extentx[0]]) #-2000,0 ## -1500,1500
            ax2.set_xlim([extentx[0],extentx[1]])
            
            for ax in [ax1,ax2]:
                ax.set_ylim([extenty[0],extenty[1]])
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(-lstep1.xmark[tru.xlines[l]],
                            lstep1.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(-lstep1.xmark[tru.ylines[l]],
                             lstep1.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                             
            for u in range(mod2.tracer_numu):
                tru=mod2.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax2.plot(lstep2.xmark[tru.xlines[l]],
                            lstep2.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax2.plot(lstep2.xmark[tru.ylines[l]],
                             lstep2.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)

        else:
            mm1 = path1.split('/')[-2]
            mm2 = path2.split('/')[-2]
            
            os.chdir(pathdata + mm1 + '/final/')
            
            #get final rim-to-rim diameter 1
            __, __, __, __, __, Drf1, __  = np.loadtxt(mm1 + '_final.txt',delimiter='\t',comments='#')
            rinner1 = Drf1 /2.
    
            os.chdir(pathdata + mm2 + '/final/')
            __, __, __, __, __, Drf2, __   = np.loadtxt(mm2 + '_final.txt',delimiter='\t',comments='#')
            rinner2 = Drf2 /2.
            
            ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,lstep1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax2.pcolormesh(mod2.x/rinner2,mod2.y/rinner2,lstep2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            
            ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
            ax2.set_xlim([0,1.2])
        
            for ax in [ax1,ax2]:
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                ax.set_ylim([-0.8,0.8])
 
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(-lstep1.xmark[tru.xlines[l]]/rinner1,
                            lstep1.ymark[tru.xlines[l]]/rinner1,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(-lstep1.xmark[tru.ylines[l]]/rinner1,
                             lstep1.ymark[tru.ylines[l]]/rinner1,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                             
            for u in range(mod2.tracer_numu):
                tru=mod2.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax2.plot(lstep2.xmark[tru.xlines[l]]/rinner2,
                            lstep2.ymark[tru.xlines[l]]/rinner2,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax2.plot(lstep2.xmark[tru.ylines[l]]/rinner2,
                             lstep2.ymark[tru.ylines[l]]/rinner2,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)        
                
    if transient:
        
        # we add the directory pathtr only in case where the transient crater was not
        # measured properly in the current modelcase (because of too coarse saving timesteps)
                
        if (pathtr != pathdata):
            mtmp0 = path1.split('/')[-2]
            mtmp1 = mtmp0.split('_L')[0]
            mtmp2 = mtmp1.split('_')
            mm1 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
            
            mtmp0 = path2.split('/')[-2]
            mtmp1 = mtmp0.split('_L')[0]
            mtmp2 = mtmp1.split('_')
            mm2 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
        else:
            mm1 = path1.split('/')[-2]
            mm2 = path2.split('/')[-2]
                
        os.chdir(pathtr + mm1 + '/transient/')
        
        #load data for the transient crater 1
        t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr1, Vr_tr, __  = np.loadtxt(mm1 + '_tr.txt',delimiter='\t',comments='#')
        data_tr1 = np.loadtxt(mm1 + '_XYtransientprofile.txt',delimiter='\t',comments='#')
        
        # plotting of the transient crater (it would be nice to have it until drim_tr)        
        mm2 = path2.split('/')[-2]
        os.chdir(pathtr + mm2 + '/transient/')
        
        #load data for the transient crater 2
        t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr2, Vr_tr, __  = np.loadtxt(mm2 + '_tr.txt',delimiter='\t',comments='#')
        data_tr2 = np.loadtxt(mm2 + '_XYtransientprofile.txt',delimiter='\t',comments='#')   
               
        #Only show the transient crater profile for height lower than 500 m over the
        # pre-impact surface
        ij1 = np.where((data_tr1[:,1]<=500.) & (data_tr1[:,0]<=Dr_tr1/2.))
        ij2 = np.where((data_tr2[:,1]<=500.) & (data_tr2[:,0]<=Dr_tr2/2.))
        
        if isinstance(norm, basestring):
            ax2.plot(data_tr2[ij2[0],0]/rinner1,data_tr2[ij2[0],1]/rinner1,color='b',linewidth=4,zorder=5)
            ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
            #plot boundary
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,lstep1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,lstep2.cmc[0],1,colors='r',linewidths=4,zorder=4)
        elif norm == 0:
            ax2.plot(data_tr2[ij2[0],0],data_tr2[ij2[0],1],color='b',linewidth=4,zorder=5)
            ax1.plot(-data_tr1[ij1[0],0],data_tr1[ij1[0],1],color='b',linewidth=4,zorder=5)           
            #plot boundary
            ax1.contour(-mod1.xc,mod1.yc,lstep1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc,mod2.yc,lstep2.cmc[0],1,colors='r',linewidths=4,zorder=4)
            
        else:
            ax2.plot(data_tr2[ij2[0],0]/rinner2,data_tr2[ij2[0],1]/rinner2,color='b',linewidth=4,zorder=5)
            ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,lstep1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,lstep2.cmc[0],1,colors='r',linewidths=4,zorder=4)
    else:
        if isinstance(norm, basestring):
            #plot boundary
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,lstep1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,lstep2.cmc[0],1,colors='r',linewidths=4,zorder=4)
        elif norm == 0:
            #plot boundary
            ax1.contour(-mod1.xc,mod1.yc,lstep1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc,mod2.yc,lstep2.cmc[0],1,colors='r',linewidths=4,zorder=4)           
        else:
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,lstep1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,lstep2.cmc[0],1,colors='r',linewidths=4,zorder=4)
    
    ax2.yaxis.tick_right()       
    if black == 1:
        for ax in [ax1,ax2]:
            ax.tick_params(axis='x', colors='white',labelsize=18)
            ax.tick_params(axis='y', colors='white',labelsize=18)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.set_axis_bgcolor('black')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
        ax1.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        
    if (path1 != path2):    
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.06, hspace=0.0)
        if black == 1:
            ax1.set_title(modelname1 + " , t = " + str(t1) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
            ax2.set_title(modelname2 + " , t = " + str(t2) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
        else:
            ax1.set_title(modelname1 + " , t = " + str(t1) + " s",position=(0.5, 0.9),fontsize=18)
            ax2.set_title(modelname2 + " , t = " + str(t2) + " s",position=(0.5, 0.9),fontsize=18)
    else:
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('right')
        ax2.xaxis.set_ticks_position('bottom')
        
        if black == 1:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.00, hspace=0.0)       
            fig.suptitle(modelname1 + " , t = " + str(t1) + " s",color = 'white',position=(0.5, 0.9),fontsize=24)
        else:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.00, hspace=0.0)       
            fig.suptitle(modelname1 + " , t = " + str(t1) + " s", position=(0.5, 0.9), fontsize=24)

    figtitle= figname + ".png"
    
    if black == 1:
        fig.savefig(paths+figtitle,dpi=300,facecolor=fig.get_facecolor(), transparent=True)
    else:
        fig.savefig(paths+figtitle,dpi=300)
    #plt.close()
    
    mod1.closeFile()
    mod2.closeFile()
        
    return fig
    

***********************************************************************

    
def evolution(path1,path2,pathdata,pathtr,paths,figname,
              modelname1,modelname2,extentx,extenty,black,norm,transient):
    
    ###
    description:
    routine that plots both the transient and final crater dimensions for 
    two different modelcases (modelcase 1 is shown in the left panel and 
    modelcase 2 is shown in the right panel).
    
    This function is pratical if for example the same 
    
    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)
    
    folders: all modelcases that need to be plotted
    
    pathplots: saving plot directory
    
    example:
    
    path1 = '/var/tmp/mysshfs/stallo/collapse/ART3/results/C00P30F06_L600/'
    path2 = '/var/tmp/mysshfs/stallo/collapse/ART3/results/C00P30F06_L600/'
    pathdata = '/media/nilscp/Zell/Collapse/data/'
    pathtr = pathdata #or
    #pathtr = '/media/nilscp/Zell/Collapse/data_transient/'
    paths="/work/nilscp/tmp/"
    
    figname = ''
    modelname1= r'C1-3, L = 600'
    modelname2= r'C1-3, L = 600'
    extentx=[0,4000]
    extenty=[-3000,1000]
    #norm = '/media/nilscp/Zell/Collapse/data/C00P30F06_L600/final/C00P30F06_L600_final.txt'
    norm = 2
    transient = True
    
    black = 1 # 1 (black) or 0 (white)
    norm = 0 # 0 (no normalization), normalization to a specific case (string to final .txt file),
    # any other numbers (normalization to the rim-to-rim crater diameter )
    
    evolution(path1,path2,pathdata,pathtr,paths,figname,
                             modelname1,modelname2,extentx,extenty,black,norm,transient)
    
    # Figure
    black = 1 # figure in black
    black = 0 # figure in white
    
    # norm
    norm = 0 nothing (normal in meters)
    norm = 1 to Drim itself
    norm = string to a specific case

    
    # if you want to plot the transient crater or not
    transient = True
    transient = False 
        
    ###
    # check if path1 and path2 are similar, if yes we only load one of the path
    if path1 == path2:
        os.chdir(path1)
        mod1 = psp.opendatfile('jdata.dat')
        mod2 = psp.opendatfile('jdata.dat')
        mm1 = path1.split('/')[-2]
        mm2 = path2.split('/')[-2]
        
    else:
           
        # load data for modelcase 1
        os.chdir(path1)   
        mod1=psp.opendatfile('jdata.dat')   
        
        # load data for modelcase 2
        os.chdir(path2)   
        mod2=psp.opendatfile('jdata.dat')
        
        mm1 = path1.split('/')[-2]
        mm2 = path2.split('/')[-2]
            
    n1 = mod1.nsteps
    n2 = mod2.nsteps
    
    if transient:
        
        # we add the directory pathtr only in case where the transient crater was not
        # measured properly in the current modelcase (because of too coarse saving timesteps)
                
        if (pathtr != pathdata):
            mtmp0 = path1.split('/')[-2]
            mtmp1 = mtmp0.split('_L')[0]
            mtmp2 = mtmp1.split('_')
            mm1 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
            
            mtmp0 = path2.split('/')[-2]
            mtmp1 = mtmp0.split('_L')[0]
            mtmp2 = mtmp1.split('_')
            mm2 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
        else:
            mm1 = path1.split('/')[-2]
            mm2 = path2.split('/')[-2]
                
        os.chdir(pathtr + mm1 + '/transient/')
        
        #load data for the transient crater 1
        t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr1, Vr_tr, __  = np.loadtxt(mm1 + '_tr.txt',delimiter='\t',comments='#')
        data_tr1 = np.loadtxt(mm1 + '_XYtransientprofile.txt',delimiter='\t',comments='#')
        
        # plotting of the transient crater (it would be nice to have it until drim_tr)        
        mm2 = path2.split('/')[-2]
        os.chdir(pathtr + mm2 + '/transient/')
        
        #load data for the transient crater 2
        t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr2, Vr_tr, __  = np.loadtxt(mm2 + '_tr.txt',delimiter='\t',comments='#')
        data_tr2 = np.loadtxt(mm2 + '_XYtransientprofile.txt',delimiter='\t',comments='#')   
    
    if n1 >= n2:
        
        for p in range(mod1.nsteps):
            
            # each timestep are saved in the variable step
            den = mod1.readStep('Den',p) 
            step1 = copy.deepcopy(den)
            
            # time in crater evolution
            time1 = np.around(step1.time,decimals=1)
            
            den2 = mod2.readStep('Den',p) 
            step2 = copy.deepcopy(den2)
            
            # time in crater evolution
            time2 = np.around(step2.time,decimals=1)
    
            # plotting of the data
            if black == 1:
                fig=plt.figure(figsize=(12,6),facecolor='black')
            else:
                fig=plt.figure(figsize=(12,6))
            
            ax1=fig.add_subplot(121)
            ax2=fig.add_subplot(122)
            
            if isinstance(norm, basestring):
                        
                #get final rim-to-rim diameter 1
                __, __, __, __, __, Drf1, __  = np.loadtxt(norm,delimiter='\t',comments='#')
                rinner1 = Drf1 /2.
                
                ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                ax2.pcolormesh(mod2.x/rinner1,mod2.y/rinner1,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                
                ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
                ax2.set_xlim([0,1.2])
            
                for ax in [ax1,ax2]:
                    ax.minorticks_off()
                    ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                    ax.set_ylim([-0.8,0.8])
                    
                for u in range(mod1.tracer_numu):
                    tru=mod1.tru[u]
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.xlines),20):
                        ax1.plot(-step1.xmark[tru.xlines[l]]/rinner1,
                                step1.ymark[tru.xlines[l]]/rinner1,
                                c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                    
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.ylines),20):
                        ax1.plot(-step1.xmark[tru.ylines[l]]/rinner1,
                                 step1.ymark[tru.ylines[l]]/rinner1,
                                 c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                 
                for u in range(mod2.tracer_numu):
                    tru=mod2.tru[u]
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.xlines),20):
                        ax2.plot(step2.xmark[tru.xlines[l]]/rinner1,
                                step2.ymark[tru.xlines[l]]/rinner1,
                                c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                    
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.ylines),20):
                        ax2.plot(step2.xmark[tru.ylines[l]]/rinner1,
                                 step2.ymark[tru.ylines[l]]/rinner1,
                                 c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            else:        
                if norm == 0:
                    ax1.pcolormesh(-mod1.x,mod1.y[::-1],step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    ax2.pcolormesh(mod2.x,mod2.y,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    
                    ax1.set_xlim([-extentx[1],extentx[0]]) #-2000,0 ## -1500,1500
                    ax2.set_xlim([extentx[0],extentx[1]])
                    
                    for ax in [ax1,ax2]:
                        ax.set_ylim([extenty[0],extenty[1]])
                        ax.minorticks_off()
                        ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                        
                    for u in range(mod1.tracer_numu):
                        tru=mod1.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax1.plot(-step1.xmark[tru.xlines[l]],
                                    step1.ymark[tru.xlines[l]],
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax1.plot(-step1.xmark[tru.ylines[l]],
                                     step1.ymark[tru.ylines[l]],
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                     
                    for u in range(mod2.tracer_numu):
                        tru=mod2.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax2.plot(step2.xmark[tru.xlines[l]],
                                    step2.ymark[tru.xlines[l]],
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax2.plot(step2.xmark[tru.ylines[l]],
                                     step2.ymark[tru.ylines[l]],
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
        
                else:
                    mm1 = path1.split('/')[-2]
                    mm2 = path2.split('/')[-2]
                    
                    os.chdir(pathdata + mm1 + '/final/')
                    
                    #get final rim-to-rim diameter 1
                    __, __, __, __, __, Drf1, __  = np.loadtxt(mm1 + '_final.txt',delimiter='\t',comments='#')
                    rinner1 = Drf1 /2.
            
                    os.chdir(pathdata + mm2 + '/final/')
                    __, __, __, __, __, Drf2, __   = np.loadtxt(mm2 + '_final.txt',delimiter='\t',comments='#')
                    rinner2 = Drf2 /2.
                    
                    ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    ax2.pcolormesh(mod2.x/rinner2,mod2.y/rinner2,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    
                    ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
                    ax2.set_xlim([0,1.2])
                
                    for ax in [ax1,ax2]:
                        ax.minorticks_off()
                        ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                        ax.set_ylim([-0.8,0.8])
         
                    for u in range(mod1.tracer_numu):
                        tru=mod1.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax1.plot(-step1.xmark[tru.xlines[l]]/rinner1,
                                    step1.ymark[tru.xlines[l]]/rinner1,
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax1.plot(-step1.xmark[tru.ylines[l]]/rinner1,
                                     step1.ymark[tru.ylines[l]]/rinner1,
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                     
                    for u in range(mod2.tracer_numu):
                        tru=mod2.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax2.plot(step2.xmark[tru.xlines[l]]/rinner2,
                                    step2.ymark[tru.xlines[l]]/rinner2,
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax2.plot(step2.xmark[tru.ylines[l]]/rinner2,
                                     step2.ymark[tru.ylines[l]]/rinner2,
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)        
                        
            if transient:
                                  
                #Only show the transient crater profile for height lower than 500 m over the
                # pre-impact surface
                ij1 = np.where((data_tr1[:,1]<=500.) & (data_tr1[:,0]<=Dr_tr1/2.))
                ij2 = np.where((data_tr2[:,1]<=500.) & (data_tr2[:,0]<=Dr_tr2/2.))
                
                if isinstance(norm, basestring):
                    ax2.plot(data_tr2[ij2[0],0]/rinner1,data_tr2[ij2[0],1]/rinner1,color='b',linewidth=4,zorder=5)
                    ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
                    #plot boundary
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
                elif norm == 0:
                    ax2.plot(data_tr2[ij2[0],0],data_tr2[ij2[0],1],color='b',linewidth=4,zorder=5)
                    ax1.plot(-data_tr1[ij1[0],0],data_tr1[ij1[0],1],color='b',linewidth=4,zorder=5)           
                    #plot boundary
                    ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    
                else:
                    ax2.plot(data_tr2[ij2[0],0]/rinner2,data_tr2[ij2[0],1]/rinner2,color='b',linewidth=4,zorder=5)
                    ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
            else:
                if isinstance(norm, basestring):
                    #plot boundary
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
                elif norm == 0:
                    #plot boundary
                    ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)           
                else:
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
            
            ax2.yaxis.tick_right()       
            if black == 1:
                for ax in [ax1,ax2]:
                    ax.tick_params(axis='x', colors='white',labelsize=18)
                    ax.tick_params(axis='y', colors='white',labelsize=18)
                    ax.spines['bottom'].set_color('white')
                    ax.spines['top'].set_color('white')
                    ax.set_axis_bgcolor('black')
                    ax.spines['left'].set_color('white')
                    ax.spines['right'].set_color('white')
                    
                ax1.spines['left'].set_color('white')
                ax2.spines['right'].set_color('white')
                
            if (path1 != path2):    
                fig.tight_layout()
                fig.subplots_adjust(wspace=0.06, hspace=0.0)
                if black == 1:
                    ax1.set_title(modelname1 + " , t = " + str(time1) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
                    ax2.set_title(modelname2 + " , t = " + str(time2) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
                else:
                    ax1.set_title(modelname1 + " , t = " + str(time1) + " s",position=(0.5, 0.9),fontsize=18)
                    ax2.set_title(modelname2 + " , t = " + str(time2) + " s",position=(0.5, 0.9),fontsize=18)
            else:
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.set_ticks_position('left')
                ax1.xaxis.set_ticks_position('bottom')
                ax2.yaxis.set_ticks_position('right')
                ax2.xaxis.set_ticks_position('bottom')
                
                if black == 1:
                    fig.tight_layout()
                    fig.subplots_adjust(wspace=0.00, hspace=0.0)       
                    fig.suptitle(modelname1 + " , t = " + str(time1) + " s",color = 'white',position=(0.5, 0.9),fontsize=24)
                else:
                    fig.tight_layout()
                    fig.subplots_adjust(wspace=0.00, hspace=0.0)       
                    fig.suptitle(modelname1 + " , t = " + str(time1) + " s", position=(0.5, 0.9), fontsize=24)
        
            if black == 1:
                figtitle= mm1 + "_" + mm2 + "_" + str(p).zfill(3) + ".png"
                fig.savefig(paths+figtitle,dpi=300,facecolor=fig.get_facecolor(), transparent=True)
                plt.close()
            else:
                figtitle= mm1 + "_" + mm2 + "_" + str(p).zfill(3) + ".png"
                fig.savefig(paths+figtitle,dpi=300)
                plt.close()
            #plt.close()
    
    elif n2 > n1:
        for p in range(mod2.nsteps):
            
            # each timestep are saved in the variable step
            den = mod1.readStep('Den',p) 
            step1 = copy.deepcopy(den)
            
            # time in crater evolution
            time1 = np.around(step1.time,decimals=1) 
            
            den2 = mod2.readStep('Den',p) 
            step2 = copy.deepcopy(den2)
            
            # time in crater evolution
            time2 = np.around(step2.time,decimals=1) 
    
            # plotting of the data
            if black == 1:
                fig=plt.figure(figsize=(12,6),facecolor='black')
            else:
                fig=plt.figure(figsize=(12,6))
            
            ax1=fig.add_subplot(121)
            ax2=fig.add_subplot(122)
            
            if isinstance(norm, basestring):
                        
                #get final rim-to-rim diameter 1
                __, __, __, __, __, Drf1, __  = np.loadtxt(norm,delimiter='\t',comments='#')
                rinner1 = Drf1 /2.
                
                ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                ax2.pcolormesh(mod2.x/rinner1,mod2.y/rinner1,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                
                ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
                ax2.set_xlim([0,1.2])
            
                for ax in [ax1,ax2]:
                    ax.minorticks_off()
                    ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                    ax.set_ylim([-0.8,0.8])
                    
                for u in range(mod1.tracer_numu):
                    tru=mod1.tru[u]
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.xlines),20):
                        ax1.plot(-step1.xmark[tru.xlines[l]]/rinner1,
                                step1.ymark[tru.xlines[l]]/rinner1,
                                c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                    
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.ylines),20):
                        ax1.plot(-step1.xmark[tru.ylines[l]]/rinner1,
                                 step1.ymark[tru.ylines[l]]/rinner1,
                                 c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                 
                for u in range(mod2.tracer_numu):
                    tru=mod2.tru[u]
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.xlines),20):
                        ax2.plot(step2.xmark[tru.xlines[l]]/rinner1,
                                step2.ymark[tru.xlines[l]]/rinner1,
                                c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                    
                    # Plot the tracers in horizontal lines, every 20 lines
                    for l in arange(0,len(tru.ylines),20):
                        ax2.plot(step2.xmark[tru.ylines[l]]/rinner1,
                                 step2.ymark[tru.ylines[l]]/rinner1,
                                 c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            else:        
                if norm == 0:
                    ax1.pcolormesh(-mod1.x,mod1.y[::-1],step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    ax2.pcolormesh(mod2.x,mod2.y,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    
                    ax1.set_xlim([-extentx[1],extentx[0]]) #-2000,0 ## -1500,1500
                    ax2.set_xlim([extentx[0],extentx[1]])
                    
                    for ax in [ax1,ax2]:
                        ax.set_ylim([extenty[0],extenty[1]])
                        ax.minorticks_off()
                        ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                        
                    for u in range(mod1.tracer_numu):
                        tru=mod1.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax1.plot(-step1.xmark[tru.xlines[l]],
                                    step1.ymark[tru.xlines[l]],
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax1.plot(-step1.xmark[tru.ylines[l]],
                                     step1.ymark[tru.ylines[l]],
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                     
                    for u in range(mod2.tracer_numu):
                        tru=mod2.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax2.plot(step2.xmark[tru.xlines[l]],
                                    step2.ymark[tru.xlines[l]],
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax2.plot(step2.xmark[tru.ylines[l]],
                                     step2.ymark[tru.ylines[l]],
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
        
                else:
                    mm1 = path1.split('/')[-2]
                    mm2 = path2.split('/')[-2]
                    
                    os.chdir(pathdata + mm1 + '/final/')
                    
                    #get final rim-to-rim diameter 1
                    __, __, __, __, __, Drf1, __  = np.loadtxt(mm1 + '_final.txt',delimiter='\t',comments='#')
                    rinner1 = Drf1 /2.
            
                    os.chdir(pathdata + mm2 + '/final/')
                    __, __, __, __, __, Drf2, __   = np.loadtxt(mm2 + '_final.txt',delimiter='\t',comments='#')
                    rinner2 = Drf2 /2.
                    
                    ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    ax2.pcolormesh(mod2.x/rinner2,mod2.y/rinner2,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
                    
                    ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
                    ax2.set_xlim([0,1.2])
                
                    for ax in [ax1,ax2]:
                        ax.minorticks_off()
                        ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                        ax.set_ylim([-0.8,0.8])
         
                    for u in range(mod1.tracer_numu):
                        tru=mod1.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax1.plot(-step1.xmark[tru.xlines[l]]/rinner1,
                                    step1.ymark[tru.xlines[l]]/rinner1,
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax1.plot(-step1.xmark[tru.ylines[l]]/rinner1,
                                     step1.ymark[tru.ylines[l]]/rinner1,
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                     
                    for u in range(mod2.tracer_numu):
                        tru=mod2.tru[u]
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.xlines),20):
                            ax2.plot(step2.xmark[tru.xlines[l]]/rinner2,
                                    step2.ymark[tru.xlines[l]]/rinner2,
                                    c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                        
                        # Plot the tracers in horizontal lines, every 20 lines
                        for l in arange(0,len(tru.ylines),20):
                            ax2.plot(step2.xmark[tru.ylines[l]]/rinner2,
                                     step2.ymark[tru.ylines[l]]/rinner2,
                                     c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)        
                        
            if transient:
                
                # we add the directory pathtr only in case where the transient crater was not
                # measured properly in the current modelcase (because of too coarse saving timesteps)
                        
                if (pathtr != pathdata):
                    mtmp0 = path1.split('/')[-2]
                    mtmp1 = mtmp0.split('_L')[0]
                    mtmp2 = mtmp1.split('_')
                    mm1 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
                    
                    mtmp0 = path2.split('/')[-2]
                    mtmp1 = mtmp0.split('_L')[0]
                    mtmp2 = mtmp1.split('_')
                    mm2 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
                else:
                    mm1 = path1.split('/')[-2]
                    mm2 = path2.split('/')[-2]
                        
                os.chdir(pathtr + mm1 + '/transient/')
                
                #load data for the transient crater 1
                t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr1, Vr_tr, __  = np.loadtxt(mm1 + '_tr.txt',delimiter='\t',comments='#')
                data_tr1 = np.loadtxt(mm1 + '_XYtransientprofile.txt',delimiter='\t',comments='#')
                
                # plotting of the transient crater (it would be nice to have it until drim_tr)        
                mm2 = path2.split('/')[-2]
                os.chdir(pathtr + mm2 + '/transient/')
                
                #load data for the transient crater 2
                t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr2, Vr_tr, __  = np.loadtxt(mm2 + '_tr.txt',delimiter='\t',comments='#')
                data_tr2 = np.loadtxt(mm2 + '_XYtransientprofile.txt',delimiter='\t',comments='#')   
                       
                #Only show the transient crater profile for height lower than 500 m over the
                # pre-impact surface
                ij1 = np.where((data_tr1[:,1]<=500.) & (data_tr1[:,0]<=Dr_tr1/2.))
                ij2 = np.where((data_tr2[:,1]<=500.) & (data_tr2[:,0]<=Dr_tr2/2.))
                
                if isinstance(norm, basestring):
                    ax2.plot(data_tr2[ij2[0],0]/rinner1,data_tr2[ij2[0],1]/rinner1,color='b',linewidth=4,zorder=5)
                    ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
                    #plot boundary
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
                elif norm == 0:
                    ax2.plot(data_tr2[ij2[0],0],data_tr2[ij2[0],1],color='b',linewidth=4,zorder=5)
                    ax1.plot(-data_tr1[ij1[0],0],data_tr1[ij1[0],1],color='b',linewidth=4,zorder=5)           
                    #plot boundary
                    ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    
                else:
                    ax2.plot(data_tr2[ij2[0],0]/rinner2,data_tr2[ij2[0],1]/rinner2,color='b',linewidth=4,zorder=5)
                    ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
            else:
                if isinstance(norm, basestring):
                    #plot boundary
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
                elif norm == 0:
                    #plot boundary
                    ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)           
                else:
                    ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
                    ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
            
            ax2.yaxis.tick_right()       
            if black == 1:
                for ax in [ax1,ax2]:
                    ax.tick_params(axis='x', colors='white',labelsize=18)
                    ax.tick_params(axis='y', colors='white',labelsize=18)
                    ax.spines['bottom'].set_color('white')
                    ax.spines['top'].set_color('white')
                    ax.set_axis_bgcolor('black')
                    ax.spines['left'].set_color('white')
                    ax.spines['right'].set_color('white')
                    
                ax1.spines['left'].set_color('white')
                ax2.spines['right'].set_color('white')
                
            if (path1 != path2):    
                fig.tight_layout()
                fig.subplots_adjust(wspace=0.06, hspace=0.0)
                if black == 1:
                    ax1.set_title(modelname1 + " , t = " + str(time1) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
                    ax2.set_title(modelname2 + " , t = " + str(time2) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
                else:
                    ax1.set_title(modelname1 + " , t = " + str(time1) + " s",position=(0.5, 0.9),fontsize=18)
                    ax2.set_title(modelname2 + " , t = " + str(time2) + " s",position=(0.5, 0.9),fontsize=18)
            else:
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.set_ticks_position('left')
                ax1.xaxis.set_ticks_position('bottom')
                ax2.yaxis.set_ticks_position('right')
                ax2.xaxis.set_ticks_position('bottom')
                
                if black == 1:
                    fig.tight_layout()
                    fig.subplots_adjust(wspace=0.00, hspace=0.0)       
                    fig.suptitle(modelname1 + " , t = " + str(time1) + " s",color = 'white',position=(0.5, 0.9),fontsize=24)
                else:
                    fig.tight_layout()
                    fig.subplots_adjust(wspace=0.00, hspace=0.0)       
                    fig.suptitle(modelname1 + " , t = " + str(time1) + " s", position=(0.5, 0.9), fontsize=24)
                    
            if black == 1:
                figtitle= mm1 + "_" + mm2 + "_" + str(p).zfill(3) + ".png"
                fig.savefig(paths+figtitle,dpi=300,facecolor=fig.get_facecolor(), transparent=True)
                plt.close()
            else:
                figtitle= mm1 + "_" + mm2 + "_" + str(p).zfill(3) + ".png"
                fig.savefig(paths+figtitle,dpi=300)
                plt.close()
            
    
    mod1.closeFile()
    mod2.closeFile()
        
    return fig




***********************************************************************
    

def timestep(path1,path2,pathdata,pathtr,paths,figname,
              modelname1,modelname2,extentx,extenty,black,norm,transient,stepnorm1,
              stepnorm2,merge):
    
    ###
    description:
    routine that plots both the transient and final crater dimensions for 
    two different modelcases (modelcase 1 is shown in the left panel and 
    modelcase 2 is shown in the right panel).
    
    This function is pratical if for example the same 
    
    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)
    
    folders: all modelcases that need to be plotted
    
    pathplots: saving plot directory
    
    example:
    
    # two different stepnorm, black, merge = False, 
    
    path1 = '/var/tmp/mysshfs/stallo/collapse/ART3/results/C00P30F06_L600/'
    path2 = '/var/tmp/mysshfs/stallo/collapse/ART3/results/C00P30F06_L600/'
    pathdata = '/media/nilscp/Zell/Collapse/data/'
    pathtr = pathdata 
    paths="/work/nilscp/tmp/"
    
    figname = ''
    modelname1= r'C1-3, L = 600'
    modelname2= r'C1-3, L = 600'
    extentx=[0,1]
    extenty=[-1,1]
    norm = '/media/nilscp/Zell/Collapse/data/C00P30F06_L600/final/C00P30F06_L600_final.txt'
    #norm = 2
    transient = False
    merge = False   
    black = 1 # 1 (black) or 0 (white)
    stepnorm1 = 10
    stepnorm2 = 10
    
    timestep(path1,path2,pathdata,pathtr,paths,figname,
              modelname1,modelname2,extentx,extenty,black,norm,transient,stepnorm1,
              stepnorm2,merge)    
    
    
    
    norm = 0 # 0 (no normalization), normalization to a specific case (string to final .txt file),
    # any other numbers (normalization to the rim-to-rim crater diameter )
    

    
    # Figure
    black = 1 # figure in black
    black = 0 # figure in white
    
    # norm
    norm = 0 nothing (normal in meters)
    norm = 1 to Drim itself
    norm = string to a specific case

    
    # if you want to plot the transient crater or not
    transient = True
    transient = False
    
    # stepnorm
    The time where it takes the data
    
    #merge (should merge or not)
    
    
    ## velocity
    path1 = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/velocity/results/C00P10F06LONG_U04_L500/'
    path2 = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/velocity/results/C00P10F06LONG_U08_L500/'
    pathdata = '/run/media/nilscp/Zell/velocity/data/'
    pathtr = pathdata 
    paths="/work/nilscp/data/article3/velocity_variation/"
    
    figname = 'test'
    modelname1= r'U = 1.6 km/s'
    modelname2= r'U = 6.4 km/s'
    extentx=[0,1]
    extenty=[-1,1]
    norm = '/run/media/nilscp/Zell/velocity/data/C00P10F06LONG_U64_L500/final/C00P10F06LONG_U64_L500_final.txt'
    norm = 2
    transient = False
    merge = False   
    black = 0 # 1 (black) or 0 (white)
    stepnorm1 = 10
    stepnorm2 = 10
    
    timestep(path1,path2,pathdata,pathtr,paths,figname,
              modelname1,modelname2,extentx,extenty,black,norm,transient,stepnorm1,
              stepnorm2,merge)    
    
        
    ###
    # check if path1 and path2 are similar, if yes we only load one of the path
    if path1 == path2:
        os.chdir(path1)
        mod1 = psp.opendatfile('jdata.dat')
        mod2 = psp.opendatfile('jdata.dat')
        mm1 = path1.split('/')[-2]
        mm2 = path2.split('/')[-2]
    else:    
        # load data for modelcase 1
        os.chdir(path1)   
        mod1=psp.opendatfile('jdata.dat')   
        
        # load data for modelcase 2
        os.chdir(path2)   
        mod2=psp.opendatfile('jdata.dat')
        
        mm1 = path1.split('/')[-2]
        mm2 = path2.split('/')[-2]
        
    
    if transient:
        
        # we add the directory pathtr only in case where the transient crater was not
        # measured properly in the current modelcase (because of too coarse saving timesteps)
                
        if (pathtr != pathdata):
            mtmp0 = path1.split('/')[-2]
            mtmp1 = mtmp0.split('_L')[0]
            mtmp2 = mtmp1.split('_')
            mm1 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
            
            mtmp0 = path2.split('/')[-2]
            mtmp1 = mtmp0.split('_L')[0]
            mtmp2 = mtmp1.split('_')
            mm2 = mtmp2[0] + '_TR_' + mtmp2[1] + '_L' + mtmp0.split('_L')[1]
        else:
            mm1 = path1.split('/')[-2]
            mm2 = path2.split('/')[-2]
                
    os.chdir(pathtr + mm1 + '/transient/')
    
    #load data for the transient crater 1
    t_tr1, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __  = np.loadtxt(mm1 + '_tr.txt',delimiter='\t',comments='#')
    data_tr1 = np.loadtxt(mm1 + '_XYtransientprofile.txt',delimiter='\t',comments='#')
    
    # plotting of the transient crater (it would be nice to have it until drim_tr)        
    os.chdir(pathtr + mm2 + '/transient/')
    
    #load data for the transient crater 2
    t_tr2, da_tr2, Da_tr2, V_tr2, h_tr2, Dr_tr2, Vr_tr2, __  = np.loadtxt(mm2 + '_tr.txt',delimiter='\t',comments='#')
    data_tr2 = np.loadtxt(mm2 + '_XYtransientprofile.txt',delimiter='\t',comments='#')
    

    os.chdir(pathdata + mm1 + '/final/')
    
    #get final rim-to-rim diameter 1
    __, __, __, __, __, Drf1, __  = np.loadtxt(mm1 + '_final.txt',delimiter='\t',comments='#')
    rinner1 = Drf1 /2.
    
    os.chdir(pathdata + mm1 + '/evolution/')
    tdata = np.loadtxt(mm1 + '_data.txt',delimiter='\t',comments='#')
    t1 = tdata[:,0]
    tnorm1 = t1 / t_tr1

    os.chdir(pathdata + mm2 + '/final/')
    __, __, __, __, __, Drf2, __   = np.loadtxt(mm2 + '_final.txt',delimiter='\t',comments='#')
    rinner2 = Drf2 /2.
    
    os.chdir(pathdata + mm2 + '/evolution/')
    tdata2 = np.loadtxt(mm2 + '_data.txt',delimiter='\t',comments='#')
    t2 = tdata2[:,0]
    tnorm2 = t2 / t_tr2
    
    __, stp1 = find_nearest(tnorm1,stepnorm1)
    __, stp2 = find_nearest(tnorm2,stepnorm2)
            
            
    # each timestep are saved in the variable step
    den = mod1.readStep('Den',stp1) 
    step1 = copy.deepcopy(den)
    
    # time in crater evolution
    time1 = np.around(step1.time,decimals=1)
    
    den2 = mod2.readStep('Den',stp2) 
    step2 = copy.deepcopy(den2)
    
    # time in crater evolution
    time2 = np.around(step2.time,decimals=1)

    # plotting of the data
    if black == 1:
        fig=plt.figure(figsize=(12,6),facecolor='black')
    else:
        fig=plt.figure(figsize=(12,6))
    
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    
    if isinstance(norm, basestring):
                
        ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
        ax2.pcolormesh(mod2.x/rinner1,mod2.y/rinner1,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
        
        ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
        ax2.set_xlim([0,1.2])
    
        for ax in [ax1,ax2]:
            ax.minorticks_off()
            ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
            ax.set_ylim([-0.8,0.8])
            
        for u in range(mod1.tracer_numu):
            tru=mod1.tru[u]
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.xlines),20):
                ax1.plot(-step1.xmark[tru.xlines[l]]/rinner1,
                        step1.ymark[tru.xlines[l]]/rinner1,
                        c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.ylines),20):
                ax1.plot(-step1.xmark[tru.ylines[l]]/rinner1,
                         step1.ymark[tru.ylines[l]]/rinner1,
                         c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                         
        for u in range(mod2.tracer_numu):
            tru=mod2.tru[u]
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.xlines),20):
                ax2.plot(step2.xmark[tru.xlines[l]]/rinner1,
                        step2.ymark[tru.xlines[l]]/rinner1,
                        c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.ylines),20):
                ax2.plot(step2.xmark[tru.ylines[l]]/rinner1,
                         step2.ymark[tru.ylines[l]]/rinner1,
                         c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
    else:        
        if norm == 0:
            ax1.pcolormesh(-mod1.x,mod1.y[::-1],step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax2.pcolormesh(mod2.x,mod2.y,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            
            ax1.set_xlim([-extentx[1],extentx[0]]) #-2000,0 ## -1500,1500
            ax2.set_xlim([extentx[0],extentx[1]])
            
            for ax in [ax1,ax2]:
                ax.set_ylim([extenty[0],extenty[1]])
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(-step1.xmark[tru.xlines[l]],
                            step1.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(-step1.xmark[tru.ylines[l]],
                             step1.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                             
            for u in range(mod2.tracer_numu):
                tru=mod2.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax2.plot(step2.xmark[tru.xlines[l]],
                            step2.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax2.plot(step2.xmark[tru.ylines[l]],
                             step2.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)

        else:
            ax1.pcolormesh(-mod1.x/rinner1,mod1.y[::-1]/rinner1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax2.pcolormesh(mod2.x/rinner2,mod2.y/rinner2,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            
            ax1.set_xlim([-1.2,0]) #-2000,0 ## -1500,1500
            ax2.set_xlim([0,1.2])
        
            for ax in [ax1,ax2]:
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                ax.set_ylim([-0.8,0.8])
 
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(-step1.xmark[tru.xlines[l]]/rinner1,
                            step1.ymark[tru.xlines[l]]/rinner1,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(-step1.xmark[tru.ylines[l]]/rinner1,
                             step1.ymark[tru.ylines[l]]/rinner1,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                             
            for u in range(mod2.tracer_numu):
                tru=mod2.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax2.plot(step2.xmark[tru.xlines[l]]/rinner2,
                            step2.ymark[tru.xlines[l]]/rinner2,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax2.plot(step2.xmark[tru.ylines[l]]/rinner2,
                             step2.ymark[tru.ylines[l]]/rinner2,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)        
                
    if transient:
        
        # we add the directory pathtr only in case where the transient crater was not
        # measured properly in the current modelcase (because of too coarse saving timesteps)
                               
        #Only show the transient crater profile for height lower than 500 m over the
        # pre-impact surface
        ij1 = np.where((data_tr1[:,1]<=500.) & (data_tr1[:,0]<=Dr_tr1/2.))
        ij2 = np.where((data_tr2[:,1]<=500.) & (data_tr2[:,0]<=Dr_tr2/2.))
        
        if isinstance(norm, basestring):
            ax2.plot(data_tr2[ij2[0],0]/rinner1,data_tr2[ij2[0],1]/rinner1,color='b',linewidth=4,zorder=5)
            ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
            #plot boundary
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
        elif norm == 0:
            ax2.plot(data_tr2[ij2[0],0],data_tr2[ij2[0],1],color='b',linewidth=4,zorder=5)
            ax1.plot(-data_tr1[ij1[0],0],data_tr1[ij1[0],1],color='b',linewidth=4,zorder=5)           
            #plot boundary
            ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
            
        else:
            ax2.plot(data_tr2[ij2[0],0]/rinner2,data_tr2[ij2[0],1]/rinner2,color='b',linewidth=4,zorder=5)
            ax1.plot(-data_tr1[ij1[0],0]/rinner1,data_tr1[ij1[0],1]/rinner1,color='b',linewidth=4,zorder=5)
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
    else:
        if isinstance(norm, basestring):
            #plot boundary
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner1,mod2.yc/rinner1,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
        elif norm == 0:
            #plot boundary
            ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)           
        else:
            ax1.contour(-mod1.xc/rinner1,mod1.yc/rinner1,step1.cmc[0],1,colors='r',linewidths=4,zorder=4)
            ax2.contour(mod2.xc/rinner2,mod2.yc/rinner2,step2.cmc[0],1,colors='r',linewidths=4,zorder=4)
    
    ax2.yaxis.tick_right()       
    if black == 1:
        for ax in [ax1,ax2]:
            ax.tick_params(axis='x', colors='white',labelsize=18)
            ax.tick_params(axis='y', colors='white',labelsize=18)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.set_axis_bgcolor('black')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
        ax1.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        
    if (path1 != path2):    
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.06, hspace=0.0)
        if black == 1:
            ax1.set_title(modelname1 + " , t = " + str(time1) + " s, $t/t_r$ = " + str(np.round(tnorm1[stp1],decimals=1)),color = 'white', position=(0.5, 0.9),fontsize=18)
            ax2.set_title(modelname2 + " , t = " + str(time2) + " s, $t/t_r$ = " + str(np.round(tnorm2[stp2],decimals=1)),color = 'white', position=(0.5, 0.9),fontsize=18)
        else:
            ax1.set_title(modelname1 + " , t = " + str(time1) + " s, $t/t_r$ = " + str(np.round(tnorm1[stp1],decimals=1)),position=(0.5, 0.9),fontsize=18)
            ax2.set_title(modelname2 + " , t = " + str(time2) + " s, $t/t_r$ = " + str(np.round(tnorm2[stp2],decimals=1)),position=(0.5, 0.9),fontsize=18)
    
    elif (merge and (path1 == path2)):
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('right')
        ax2.xaxis.set_ticks_position('bottom')
        
        if black == 1:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.00, hspace=0.0)       
            fig.suptitle(modelname1 + " , t = " + str(time1) + " s, $t/t_r$ = " + str(np.round(tnorm1[stp1],decimals=1)),color = 'white',position=(0.5, 0.9),fontsize=24)
        else:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.00, hspace=0.0)       
            fig.suptitle(modelname2 + " , t = " + str(time2) + " s, $t/t_r$ = " + str(np.round(tnorm2[stp2],decimals=1)),position=(0.5, 0.9),fontsize=24)
    else:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.06, hspace=0.0)
        if black == 1:
            ax1.set_title(modelname1 + " , t = " + str(time1) + " s, $t/t_r$ = " + str(np.round(tnorm1[stp1],decimals=1)),color = 'white', position=(0.5, 0.9),fontsize=18)
            ax2.set_title(modelname2 + " , t = " + str(time2) + " s, $t/t_r$ = " + str(np.round(tnorm2[stp2],decimals=1)),color = 'white', position=(0.5, 0.9),fontsize=18)
        else:
            ax1.set_title(modelname1 + " , t = " + str(time1) + " s, $t/t_r$ = " + str(np.round(tnorm1[stp1],decimals=1)),position=(0.5, 0.9),fontsize=18)
            ax2.set_title(modelname2 + " , t = " + str(time2) + " s, $t/t_r$ = " + str(np.round(tnorm2[stp2],decimals=1)),position=(0.5, 0.9),fontsize=18)
    if black == 1:
        figtitle= mm1 + "_" + mm2 + "_norm" + str(stepnorm1).zfill(3) + ".png"
        fig.savefig(paths+figtitle,dpi=300,facecolor=fig.get_facecolor(), transparent=True)
        plt.close()
    else:
        figtitle= mm1 + "_" + mm2 + "_norm" + str(stepnorm1).zfill(3) + ".png"
        fig.savefig(paths+figtitle,dpi=300)
        plt.close()
    #plt.close()
    
    mod1.closeFile()
    mod2.closeFile()
        
    return fig

   

def transient2(path1,path2,pathdata,paths,figname,
              modelname1,modelname2,extentx,extenty,black,norm,merge):
    
    ###
    description:
    routine that plots both the transient and final crater dimensions for 
    two different modelcases (modelcase 1 is shown in the left panel and 
    modelcase 2 is shown in the right panel).
    
    This function is pratical if for example the same 
    
    inputs:
    pathdata: path where the text files have been stored (text files that are obtained
    from running the main.py script)
    
    folders: all modelcases that need to be plotted
    
    pathplots: saving plot directory
    
    example:
    
    
    path = '/media/nilscp/Cloud/SENS/PORFRIC/results/'
    path1 = '/media/nilscp/Cloud/SENS/COH/results/C01P20F06_L500/'
    path2 = '/media/nilscp/Cloud/SENS/COH/results/C05P20F06_L500/'
    
    
    # two different stepnorm, black, merge = False, 
    
    pathdata = '/media/nilscp/Cloud/SENS/data/'
    paths="/work/nilscp/tmp/"   
    norm = '/media/nilscp/Cloud/SENS/data/C00P20F06_L500/transient/C00P20F06_L500_tr.txt'
    modelname1 = r'$Y_0 = 100 kPa , L = 500 m$'
    modelname2 = r'$Y_0 = 500 kPa, L = 500 m$'
    extentx = [0,0]
    extenty = [0,0]
    black = 1
    merge = False
    figname = ''
    
    transient2(path1,path2,pathdata,paths,figname,
              modelname1,modelname2,extentx,extenty,black,norm,merge)
              
    norm = 0 # 0 (no normalization), normalization to a specific case (string to final .txt file),
    # any other numbers (normalization to the rim-to-rim crater diameter )
    

    
    # Figure
    black = 1 # figure in black
    black = 0 # figure in white
    
    # norm
    norm = 0 nothing (normal in meters)
    norm = 1 to Drim itself
    norm = string to a specific case

    
    # if you want to plot the transient crater or not
    transient = True
    transient = False
    
    # stepnorm
    The time where it takes the data
    
    #merge (should merge or not)
    
        
    ###
    # check if path1 and path2 are similar, if yes we only load one of the path
    if path1 == path2:
        os.chdir(path1)
        mod1 = psp.opendatfile('jdata.dat')
        mod2 = psp.opendatfile('jdata.dat')
        mm1 = path1.split('/')[-2]
        mm2 = path2.split('/')[-2]
    else:    
        # load data for modelcase 1
        os.chdir(path1)   
        mod1=psp.opendatfile('jdata.dat')   
        
        # load data for modelcase 2
        os.chdir(path2)   
        mod2=psp.opendatfile('jdata.dat')
        
        mm1 = path1.split('/')[-2]
        mm2 = path2.split('/')[-2]
        
            
    # we add the directory pathtr only in case where the transient crater was not
    # measured properly in the current modelcase (because of too coarse saving timesteps)
    mm1 = path1.split('/')[-2]
    mm2 = path2.split('/')[-2]
            
    os.chdir(pathdata + mm1 + '/transient/')
    
    #load data for the transient crater 1
    t_tr1, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __  = np.loadtxt(mm1 + '_tr.txt',delimiter='\t',comments='#')
    Da_tr1 = Da_tr1 /2. 
    
    # plotting of the transient crater (it would be nice to have it until drim_tr)        
    os.chdir(pathdata + mm2 + '/transient/')
    
    #load data for the transient crater 2
    t_tr2, da_tr2, Da_tr2, V_tr2, h_tr2, Dr_tr2, Vr_tr2, __  = np.loadtxt(mm2 + '_tr.txt',delimiter='\t',comments='#')
    Da_tr2 = Da_tr2 /2. 
    
    os.chdir(pathdata + mm1 + '/evolution/')
    tdata = np.loadtxt(mm1 + '_data.txt',delimiter='\t',comments='#')
    t1 = tdata[:,0]
    tnorm1 = t1 / t_tr1
    
    os.chdir(pathdata + mm2 + '/evolution/')
    tdata2 = np.loadtxt(mm2 + '_data.txt',delimiter='\t',comments='#')
    t2 = tdata2[:,0]
    tnorm2 = t2 / t_tr2
    
    __, stp1 = find_nearest(tnorm1,1.0)
    __, stp2 = find_nearest(tnorm2,1.0)
            
            
    # each timestep are saved in the variable step
    den = mod1.readStep('Den',stp1) 
    step1 = copy.deepcopy(den)
    
    # time in crater evolution
    time1 = np.around(step1.time,decimals=1)
    
    den2 = mod2.readStep('Den',stp2) 
    step2 = copy.deepcopy(den2)
    
    # time in crater evolution
    time2 = np.around(step2.time,decimals=1)

    # plotting of the data
    if black == 1:
        fig=plt.figure(figsize=(12,6),facecolor='black')
    else:
        fig=plt.figure(figsize=(12,6))
    
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    
    if isinstance(norm, basestring):
        
        t_tr, da_tr, Da_tr, V_tr, h_tr, Dr_tr, Vr_tr, __  = np.loadtxt(norm,delimiter='\t',comments='#')
        Da_tr = Da_tr /2.
        
        ax1.pcolormesh(-mod1.x/Da_tr,mod1.y[::-1]/Da_tr,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
        ax2.pcolormesh(mod2.x/Da_tr,mod2.y/Da_tr,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
        
        ax1.set_xlim([-2.0,0]) #-2000,0 ## -1500,1500
        ax2.set_xlim([0,2.0])
    
        for ax in [ax1,ax2]:
            ax.minorticks_off()
            ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
            ax.set_ylim([-1.2,0.8])
            
        for u in range(mod1.tracer_numu):
            tru=mod1.tru[u]
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.xlines),20):
                ax1.plot(-step1.xmark[tru.xlines[l]]/Da_tr,
                        step1.ymark[tru.xlines[l]]/Da_tr,
                        c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.ylines),20):
                ax1.plot(-step1.xmark[tru.ylines[l]]/Da_tr,
                         step1.ymark[tru.ylines[l]]/Da_tr,
                         c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                         
        for u in range(mod2.tracer_numu):
            tru=mod2.tru[u]
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.xlines),20):
                ax2.plot(step2.xmark[tru.xlines[l]]/Da_tr,
                        step2.ymark[tru.xlines[l]]/Da_tr,
                        c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
            
            # Plot the tracers in horizontal lines, every 20 lines
            for l in arange(0,len(tru.ylines),20):
                ax2.plot(step2.xmark[tru.ylines[l]]/Da_tr,
                         step2.ymark[tru.ylines[l]]/Da_tr,
                         c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
    else:        
        if norm == 0:
            ax1.pcolormesh(-mod1.x,mod1.y[::-1],step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax2.pcolormesh(mod2.x,mod2.y,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            
            ax1.set_xlim([-extentx[1],extentx[0]]) #-2000,0 ## -1500,1500
            ax2.set_xlim([extentx[0],extentx[1]])
            
            for ax in [ax1,ax2]:
                ax.set_ylim([extenty[0],extenty[1]])
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(-step1.xmark[tru.xlines[l]],
                            step1.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(-step1.xmark[tru.ylines[l]],
                             step1.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                             
            for u in range(mod2.tracer_numu):
                tru=mod2.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax2.plot(step2.xmark[tru.xlines[l]],
                            step2.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax2.plot(step2.xmark[tru.ylines[l]],
                             step2.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)

        else:
            ax1.pcolormesh(-mod1.x/Da_tr1,mod1.y[::-1]/Da_tr1,step1.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax2.pcolormesh(mod2.x/Da_tr2,mod2.y/Da_tr2,step2.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            
            ax1.set_xlim([-2.0,0]) #-2000,0 ## -1500,1500
            ax2.set_xlim([0,2.0])
        
            for ax in [ax1,ax2]:
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                ax.set_ylim([-1.2,0.8])
 
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(-step1.xmark[tru.xlines[l]]/Da_tr1,
                            step1.ymark[tru.xlines[l]]/Da_tr1,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(-step1.xmark[tru.ylines[l]]/Da_tr1,
                             step1.ymark[tru.ylines[l]]/Da_tr1,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                             
            for u in range(mod2.tracer_numu):
                tru=mod2.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax2.plot(step2.xmark[tru.xlines[l]]/Da_tr2,
                            step2.ymark[tru.xlines[l]]/Da_tr2,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax2.plot(step2.xmark[tru.ylines[l]]/Da_tr2,
                             step2.ymark[tru.ylines[l]]/Da_tr2,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)        
                        
    # we add the directory pathtr only in case where the transient crater was not
    # measured properly in the current modelcase (because of too coarse saving timesteps)
                           
    #Only show the transient crater profile for height lower than 500 m over the
    # pre-impact surface

    if isinstance(norm, basestring):
        ax1.contour(-mod1.xc/Da_tr,mod1.yc/Da_tr,step1.cmc[0],1,colors='b',linewidths=4,zorder=4)
        ax2.contour(mod2.xc/Da_tr,mod2.yc/Da_tr,step2.cmc[0],1,colors='b',linewidths=4,zorder=4)
        ax1.hlines(0,-2,0,'r',linewidths=4,zorder=-10)
        ax2.hlines(0,0,2,'r',linewidths=4,zorder=-10)

    elif norm == 0:
        ax1.contour(-mod1.xc,mod1.yc,step1.cmc[0],1,colors='b',linewidths=4,zorder=4)
        ax2.contour(mod2.xc,mod2.yc,step2.cmc[0],1,colors='b',linewidths=4,zorder=4)
        ax1.hlines(0,-extentx[1],extentx[0],'r',linewidths=4,zorder=-10)
        ax2.hlines(0,extentx[0],extentx[1],'r',linewidths=4,zorder=-10)
        
    else:
        ax1.contour(-mod1.xc/Da_tr1,mod1.yc/Da_tr1,step1.cmc[0],1,colors='b',linewidths=4,zorder=4)
        ax2.contour(mod2.xc/Da_tr2,mod2.yc/Da_tr2,step2.cmc[0],1,colors='b',linewidths=4,zorder=4)
        ax1.hlines(0,-2,0,'r',linewidths=4,zorder=-10)
        ax2.hlines(0,0,2,'r',linewidths=4,zorder=-10)

    
    ax2.yaxis.tick_right()       
    if black == 1:
        for ax in [ax1,ax2]:
            ax.tick_params(axis='x', colors='white',labelsize=18)
            ax.tick_params(axis='y', colors='white',labelsize=18)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.set_axis_bgcolor('black')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
        ax1.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        
    if (path1 != path2):    
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.06, hspace=0.0)
        if black == 1:
            l = ax1.set_title(modelname1 + " , t = " + str(time1) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
            l2 = ax2.set_title(modelname2 + " , t = " + str(time2) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
            l.set_zorder(20)
            l2.set_zorder(20)
        else:
            l = ax1.set_title(modelname1 + " , t = " + str(time1) + " s",position=(0.5, 0.9),fontsize=18)
            l2 = ax2.set_title(modelname2 + " , t = " + str(time2) + " s",position=(0.5, 0.9),fontsize=18)
            l.set_zorder(20)
            l2.set_zorder(20)
    elif (merge and (path1 == path2)):
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('right')
        ax2.xaxis.set_ticks_position('bottom')
        
        if black == 1:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.00, hspace=0.0)       
            fig.suptitle(modelname1 + " , t = " + str(time1) + " s",color = 'white',position=(0.5, 0.9),fontsize=24)
        else:
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.00, hspace=0.0)       
            fig.suptitle(modelname2 + " , t = " + str(time2) + " s",position=(0.5, 0.9),fontsize=24)
    else:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.06, hspace=0.0)
        if black == 1:
            l = ax1.set_title(modelname1 + " , t = " + str(time1) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
            l2 = ax2.set_title(modelname2 + " , t = " + str(time2) + " s",color = 'white', position=(0.5, 0.9),fontsize=18)
            l.set_zorder(20)
            l2.set_zorder(20)
        else:
            l = ax1.set_title(modelname1 + " , t = " + str(time1) + " s" ,position=(0.5, 0.9),fontsize=18)
            l2 = ax2.set_title(modelname2 + " , t = " + str(time2) + " s" ,position=(0.5, 0.9),fontsize=18)
            l.set_zorder(20)
            l2.set_zorder(20)
            #ax1.set_zorder(100)
            #ax2.set_zorder(100)
    if black == 1:
        figtitle= mm1 + "_" + mm2 + "_transient" + ".png"
        fig.savefig(paths+figtitle,dpi=300,facecolor=fig.get_facecolor(), transparent=True)
        plt.close()
    else:
        figtitle= mm1 + "_" + mm2 + "_transient" + ".png"
        fig.savefig(paths+figtitle,dpi=300)
        plt.close()
    #plt.close()
    
    mod1.closeFile()
    mod2.closeFile()
        
    return fig



def evoplottr(path1,paths, path1tr, pathstr, norm, extentx, extenty, lbl):
    
    ###
    path1 = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/layering/AUG/collapse/results/C00P10F06LONG_L500/'
    paths = '/run/media/nilscp/Zell/Collapse/data/C00P10F06LONG_L500/'
    path1tr = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/layering/AUG/collapse/results/C00P10F07LONG_L500/'
    pathstr = '/run/media/nilscp/Zell/Collapse/data/C00P10F07LONG_L500/'
    
    
    
    path1 = '/run/media/nilscp/Cloud/SENS/COH/results/C01P10F06_L100/'
    paths = '/run/media/nilscp/Cloud/SENS/data/C01P10F06_L100/'
    path1tr =  '/run/media/nilscp/Cloud/SENS/PORFRIC/results/C00P10F06_L100/'
    pathstr = '/run/media/nilscp/Cloud/SENS/data/C00P10F06_L100/'
    norm = 2
    extentx = [0, 5000]
    extenty = [-3000,1500]
    lbl = r"$ Y_0 = 100 kPa, \mathit{L} = 100 m$"
    
    evoplottr(path1,paths, path1tr, pathstr, norm, extentx, extenty, lbl)
    ###
        
    os.chdir(path1)
    mod1 = psp.opendatfile('jdata.dat')
        
    modelname = path1.split('/')[-2]
    
    if not os.path.exists(paths + 'plots'):
        os.makedirs(paths + 'plots')
        
    if norm == 1:                
        os.chdir(paths + '/transient/')
        
        #load data for the transient crater 1
        t_trg, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __  = np.loadtxt(modelname + '_tr.txt',delimiter='\t',comments='#')
        Ra_tr1 = Da_tr1 / 2.
        
        
    elif norm == 2:
        
        os.chdir(paths + '/transient/')
        
        #load data for the transient crater 
        t_trg, __, __, __, __, __, __, __  = np.loadtxt(modelname + '_tr.txt',delimiter='\t',comments='#')
        
        
        modelnametr = pathstr.split('/')[-2]                                                
        os.chdir(pathstr + '/transient/')
                                               
        #load data for the transient crater modtr
        t_tr1, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __  = np.loadtxt(modelnametr + '_tr.txt',delimiter='\t',comments='#')
        Ra_tr1 = Da_tr1 / 2.
        
        
    else:
        os.chdir(paths + '/transient/')
        
        #load data for the transient crater 1
        t_trg, da_tr1, Da_tr1, V_tr1, h_tr1, Dr_tr1, Vr_tr1, __  = np.loadtxt(modelname + '_tr.txt',delimiter='\t',comments='#')
        
        
        os.chdir(paths + '/evolution/')
        tdata = np.loadtxt(modelname + '_data.txt',delimiter='\t',comments='#')
        t1 = tdata[:,0]
        tnorm1 = t1 / t_trg
        print np.max(tnorm1)
        __, stp1 = find_nearest(tnorm1,10)
        
    os.chdir(path1)       
    for i in range(mod1.nsteps):
        step=mod1.readStep('Den',i)
    
        t1 = step.time       
        # plotting of the data
        fig=plt.figure(figsize=(6,6))        
        ax1=fig.add_subplot(111)
        
        if norm >= 1:
            
            ax1.pcolormesh(mod1.x/Ra_tr1,mod1.y/Ra_tr1,step.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax1.contour(mod1.xc/Ra_tr1,mod1.yc/Ra_tr1,step.cmc[0],1,colors='b',linewidths=4,zorder=4)
            ax1.hlines(0,0,1.2,'r',linewidths=4,zorder=1)
            ax1.set_xlim([0,1.2]) #-2000,0 ## -1500,1500
        
            for ax in [ax1]:
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                ax.set_ylim([-0.8,0.8])
                
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(step.xmark[tru.xlines[l]]/Ra_tr1,
                            step.ymark[tru.xlines[l]]/Ra_tr1,
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(step.xmark[tru.ylines[l]]/Ra_tr1,
                             step.ymark[tru.ylines[l]]/Ra_tr1,
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                    
        else:       
            ax1.pcolormesh(mod1.x,mod1.y,step.mat,cmap='BrBG_r',vmin=-5., vmax = 5., zorder=2)
            ax1.contour(mod1.xc,mod1.yc,step.cmc[0],1,colors='b',linewidths=4,zorder=4)
            ax1.hlines(0,0,extentx[1],'r',linewidths=4,zorder=1)               
            ax1.set_xlim([extentx[0],extentx[1]]) #-2000,0 ## -1500,1500
            
            for ax in [ax1]:
                ax.set_ylim([extenty[0],extenty[1]])
                ax.minorticks_off()
                ax.tick_params('both', labelsize=16,length=10, width=2., which='major')
                
            for u in range(mod1.tracer_numu):
                tru=mod1.tru[u]
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.xlines),20):
                    ax1.plot(step.xmark[tru.xlines[l]],
                            step.ymark[tru.xlines[l]],
                            c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                
                # Plot the tracers in horizontal lines, every 20 lines
                for l in arange(0,len(tru.ylines),20):
                    ax1.plot(step.xmark[tru.ylines[l]],
                             step.ymark[tru.ylines[l]],
                             c='k',marker='.',linestyle='None',markersize=1.0,zorder=3)
                                
        if norm == 0:
            ax.set_xlabel('x (m)', fontsize = 20)
            ax.set_ylabel('y (m)', fontsize = 20)
            fig.tight_layout()
            
        else:
            ax.set_xlabel(r'$x / R_t_c$', fontsize = 20)
            ax.set_ylabel(r'$y / R_t_c$', fontsize = 20)
            fig.tight_layout()  
        
        
        ax1.set_title(r"$\mathit{t} = " + str(np.around(t1,decimals=2)) + " " +" s, $" + " " + "$\zeta$ = " + str(np.around(t1/t_trg,decimals=2)),position=(0.5, 0.9),fontsize=18)
        st = fig.suptitle(lbl, fontsize=22)
        st.set_y(0.97)
        st.set_x(0.55)
        fig.subplots_adjust(top=0.9)
        
        if norm == 0:
            figtitle= modelname + '_' + str(int(i)).zfill(3)+ ".png"
            fig.savefig(paths+ 'plots/' + figtitle,dpi=300)
        else:
            figtitle= 'norm_' + modelname + '_' + str(int(i)).zfill(3)+ ".png"
            fig.savefig(paths+ 'plots/' + figtitle,dpi=300)
        plt.close()
        
    mod1.closeFile()
            



***********************************************************************
'''
