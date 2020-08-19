# -*- coding: utf-8 -*-

'''
******************************************************************************

     =========================================================================
     Subroutine to calculate crater dimensions at each step and for the transient
     and final crater diameters. 


     Called from xxx

     Description                                     Programmer    Date
     ------------------------------------------------------------------
     Original version (1.0).............................NCP  2016/01/12
     Improved version (2.0)   ..........................NCP  2017/22/11   
    ==========================================================================
    
    The version 2.0 includes:
    - a better description of functions
    - changed the name of some function (except main and craterdimensions)
    - functions:
    
    movingaverage: average values along a moving window 
    ixdiam: first non-empty cell along the pre-impact surface (x)
    appvolume: apparent crater volume
    rimvolume: crater volume up to the rim crest
    maxappdepth: maximum apparent crater depth
    midappdepth: apparent crater depth along the axis of symmetry
    appdiameter: apparent crater diameter
    emptycells: calculate the number of empty cells
    rimrim: calculate rim height, rim-to-rim diameter and x- and y-indexes
    craterdimensions: calculate all final crater dimensions for 1-layer targets (each step)
    main: main python file, run craterdimensions and save to txt file
    rimrimProfile: crater profile up to the crest rim
    craterProfiles: Derive crater profile for mode = 1 (transient), 2 (final), 3 (arbitrary)
    craterProfileXY: Derive crater profile for chosen saved timestep and layer (id_mat)
    tokenize: function to sort files in function of number in the filenames
    morph2layers: calculate crater dimensions and profiles for 2-layer targets
    
    ==========================================================================
    
    I should give the possibility to define the transient crater diameter at the maximum depth during
    crater evolution. So we can choose for example this mode in the case of low velocity impact and problem related
    to the somehow sinking of the impact structure.

******************************************************************************
'''

# Basic modules are loaded
import numpy as np
import os
import sys
import copy

# path to in-house scripts and pySALEPlot
path_pySALEPlot = '/work/nilscp/iSALE/Dellen/lib'

# path to pySALEPlot is loaded
sys.path.append(path_pySALEPlot)
import pySALEPlot as psp

'''
***********************************************************************
'''


def movingaverage(values, window):
    '''
    description:
    average values along a moving window

    example:
    X = np.linspace(1,100,1000)
    avg_value_X = movingaverage(X,20) #average over 20 

    '''

    weights = np.repeat(1.0, window)/window
    ma = np.convolve(values, weights, 'valid')
    return ma


'''
***********************************************************************
'''


def ixdiam(step, id_mat, oridy):
    '''
    description:
    return the first non-empty cell along the pre-impact surface (x)

    example:


    # get the y-index that correponds to the pre-impact surface
    oridy = np.where(model.yy == 0.)[0][0]

    #load the model
    model = psp.opendatfile(path + modelname + '/jdata.dat')    

    #load data at the 10th saved timestep
    den = model.readStep('Den',0) 

    #copy data
    step = copy.deepcopy(den)

    #id_mat
    id_mat = 0 #(all) other type of materials can be selected as well (1,2,..)

    # return the index
    ix = ixdiam(step,0,oridy)


    '''
    # get the concentration of material in a cell along the pre-impact surface
    conc = step.cmc[id_mat][:, oridy]

    # get a moving average over 20 cells (to avoid detection of ejected materials)
    ma = movingaverage(conc, 20)

    # if the amount of material fill more than 99.9% of the cell, get the index
    if np.max(ma > 0.999):
        ix = np.where(ma > 0.999)[0][0]
    # otherwise get the index where the first maximum is detected
    else:
        ix = np.where(ma == np.max(ma))[0][-1]

    return ix


'''
***********************************************************************
'''


def appvolume(model, step, time, id_mat, oridy, diam, ix):
    '''
    description:
    calculate the volume of the apparent crater diameter (from the pre-impact
    surface, Va)

    inputs:
    model: model output from psp
    step: step loaded from psp
    time: timestep
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)
    oridy: y-index pre-impact surface
    ix: first non-empty cell along the pre-impact surface (x)

    '''
    # if this is the first time step, the volume is then equal to 0
    if time == 0:
        Vtot = 0
    elif np.isnan(diam):
        Vtot = np.nan
    else:
        # get the concentration in all cells up to the pre-impact surface (y)
        # and up to the apparent crater diameter (x)
        conc = step.cmc[id_mat][:ix, :oridy]

        # select only cells that are not completelly filled with materials
        empty_cells = np.where(conc < 1)

        # select the x-index, x-index + 1 and y-index
        i = empty_cells[0]
        ii = empty_cells[0] + 1
        j = empty_cells[1]

        # r is the distance from the axis of symmetry
        r = model.x[i, j]

        # h is the height of the cells, which is the same for all the cells
        h = model.dy

        # the distance along the x-axis between two cells
        # takes into account the concentration of materials
        R = model.x[i, j] + ((1 - step.cmc[id_mat][i, j])
                             * (model.x[ii, j] - model.x[i, j]))

        # calculation of the volume (volume of a cylinder)
        vol = np.pi * h * ((R**2) - (r**2))
        V = np.sum(vol)
        Vtot = V

    return Vtot


'''
***********************************************************************
'''


def rimvolume(model, step, time, id_mat, diam, ix, iy):
    '''
    description:
    calculate the volume of the crater diameter up to the rim crest, Vr

    this function is similar to appvolume except that the volume is calculated
    up to the rim, which is defined with the indexs ix and iy, from function 
    rimrim (3rd and last outputs)

    inputs:
    model: model output from psp
    step: step loaded from psp
    time: timestep
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)
    oridy: y-index pre-impact surface
    ix: first non-empty cell along iy (x)
    iy: y-index where the rim is detected

    '''
    # if this is the first time step, the volume is then equal to 0
    if time == 0:
        Vtot = 0
    elif ((np.isnan(diam)) or (np.isnan(ix)) or (np.isnan(iy))):
        Vtot = np.nan
    else:
        # get the concentration in all cells up to the rim height (y)
        # and up to the rim-rim crater diameter (x)
        conc = step.cmc[id_mat][:ix, :iy]

        # select only cells that are not completelly filled with materials
        empty_cells = np.where(conc < 1)  # here I have the index

        # select the x-index, x-index + 1 and y-index
        i = empty_cells[0]
        ii = empty_cells[0] + 1
        j = empty_cells[1]

        # r is the distance from the axis of symmetry
        r = model.x[i, j]

        # h is the height of the cells, which is the same for all the cells
        h = model.dy

        # the distance along the x-axis between two cells
        # takes into account the concentration of materials
        R = model.x[i, j] + ((1 - step.cmc[id_mat][i, j])
                             * (model.x[ii, j] - model.x[i, j]))

        # calculation of the volume (volume of a cylinder)
        vol = np.pi * h * ((R**2) - (r**2))
        V = np.sum(vol)
        Vtot = V

    return Vtot


'''
***********************************************************************
'''


def maxappdepth(model, step, time, id_mat, oridy, diam, ix):
    '''
    description:
    calculate the maximum apparent crater depth (da)
    from the pre-impact surface

    inputs:
    model: model output from psp
    step: saved timestep loaded from psp
    time: time of saved timestep
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)
    oridy: y-index pre-impact surface
    ix: first non-empty cell along the pre-impact surface (x)


    '''

    # if the simulation did not start (t = 0) or the apparent crater diameter
    # is not detected ix = 0, then the max app depth is equal to 0.
    if (time == 0) or (ix == 0):
        depth = 0
    elif np.isnan(diam):
        depth = np.nan
    else:
        # store how much the cells are filled with materials
        conc = step.cmc[id_mat][:ix, :oridy]

        # select cells, which are partially filled
        empty_cells = np.where((conc > 0) & (
            conc < 1))  # here I have the index

        # if there are no partially filled cells then depth is equal to 0
        # I guess this condition is not likely to be True
        if empty_cells[0].size == 0:
            depth = np.nan

        else:
            # get x- and y-index
            i = empty_cells[0]
            j = empty_cells[1]

            # create empty new x- and y-index
            ik = np.array([])
            jk = np.array([])

            for idx, values in np.ndenumerate(j):

                # a threshold is required
                # we use a 20-cell window to exclude possible ejected materials
                thres_d = np.nanmean(step.cmc[0][i[idx[0]], values-20:values])

                # if the 20 cells below the selected cells have more than 99%
                # of materials in average, then i and j are added to ik and jk
                if (thres_d > 0.99):

                    jk = np.append(jk, values)
                    ik = np.append(ik, i[idx[0]])

            # the index where the maximum depth is reached can now be selected
            idx_depth = np.where(jk == np.min(jk))[0][0]

            # conversion to integer
            iif = ik[idx_depth].astype(int)
            ijf = jk[idx_depth].astype(int)

            # the maximum apparent depth is calculated, taking into account
            # the concentration of materials in the cell.
            depth = np.abs((model.y[iif, ijf+1]) -
                           ((1 - step.cmc[0][iif, ijf]) * model.dy))

    return depth


'''
***********************************************************************
'''


def midappdepth(model, step, time, id_mat, diam):
    '''
    description:
    calculate the maximum apparent crater depth along the axis of symmetry (da)
    from the pre-impact surface

    inputs:
    model: model output from psp
    step: saved timestep loaded from psp
    time: time of saved timestep
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)
    '''

    # if first time step, then the depth is obviously equal to 0
    if time == 0:
        depth = 0
    elif np.isnan(diam):
        depth = np.nan
    else:
        # the concentration of materials is only check along the axis of symmetry
        conc = step.cmc[id_mat][0, :]

        # a 20-cell window average is used
        ma = movingaverage(conc, 20)

        # as low indexes represent the deep material, we here use a threshold
        # where the boundary is detected when less than 1% of material is filled
        # in average over the 20-cell average (not the best way)
        ijf = np.where(ma < 0.01)[0][0]

        # the depth is then taken
        depth = np.abs((model.y[0, ijf]) -
                       ((1 - step.cmc[0][0, ijf-1]) * model.dy))

    return depth


'''
***********************************************************************
'''


def appdiameter(model, step, time, oridy, ix):
    '''
    description:
    calculate the apparent crater diameter along the pre-impact surface (Da)

    inputs:
    model: model output from psp
    step: saved timestep loaded from psp
    time: time of saved timestep
    oridy: y-index pre-impact surface
    ix: first non-empty cell along the pre-impact surface (x)    
    '''

    # if first time step, then the diameter is obviously equal to 0
    if time == 0:
        diam = 0
    else:
        # the boundary of the crater is previously detected in function ixdiam
        # the concentration of material(s) in the cell is here calculated
        cmc_diam = (1. - step.cmc[0][ix-1, oridy]) * model.dx

        # the apparent crater diameter is then calculated
        diam = (np.abs(model.x[ix-1, oridy]) + cmc_diam) * 2.

        # if the apparent crater diameter calculated is larger than the
        # hi resolution than set value equal to 0
        if diam > (model.xhires[1] * 2.):
            diam = np.nan  # overwrite previously calculated value

    return diam


'''
***********************************************************************
'''


def emptycells(model, step, time, oridy, diam, ix):
    '''
    description:
    calculate the number of empty cells below the pre-impact surface

    inputs:
    model: model output from psp
    step: saved timestep loaded from psp
    time: time of saved timestep
    oridy: y-index pre-impact surface
    ix: first non-empty cell along the pre-impact surface (x)

    outputs:
    ec: number of empty cells (conc = 0)
    ls: lower than 1 (conc < 1)
    pf: partially filled (0 < conc < 1)

    '''
    #
    if time == 0:
        ec = 0  # empty cells
        pf = 0  # partially filled
        ls = 0  # less than 1
    elif np.isnan(diam):
        ec = np.nan
        pf = np.nan
        ls = np.nan
    else:
        sl = step.cmc[0][0:ix+1, :oridy]
        ec = np.sum((sl == 0))
        # sum of concentration of matter
        pf = np.sum(sl[np.where((sl > 0) & (sl < 1))])
        ls = np.sum(sl < 1)

    return ec, ls, pf


'''
***********************************************************************
'''


def rimrim(model, step, time, id_mat, oridy, diam, ix):
    ''' 
    description:
    calculate crater dimensions up to the rim crest

    inputs:
    model: model output from psp
    step: saved timestep loaded from psp
    time: time of saved timestep
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)
    oridy: y-index pre-impact surface
    ix: first non-empty cell along the pre-impact surface (x)

    outputs:
    alt: Elevation (y-axis), cross section profile
    drim: Distance (x-axis), cross section profile
    iif: index of rim-to-rim crater diameter (x-index)
    ijf: index of rim-to-rim crest height (y-index)

    iif and ijf can be used in the function "rimvolume"

    '''
    # if time equal to 0, then all crater dimensions and indexes are equal to 0
    if time == 0:
        alt = 0
        drim = 0
        iif = 0
        ijf = 0

    elif np.isnan(diam):
        alt = np.nan
        drim = np.nan
        iif = np.nan
        ijf = np.nan

    else:
        # we neglect the 50 closest cells to the axis of symmetry
        conc = step.cmc[id_mat][:-50, :]

        # and we take only the partially filled cells (conc = concentration of
        # matters)
        empty_cells = np.where((conc > 0) & (conc < 1))

        # x- and y-indexes
        i = empty_cells[0]
        j = empty_cells[1]

        # creation of new indexes
        ik = np.array([])
        jk = np.array([])

        for idx, values in np.ndenumerate(j):

            # we loop through every index, and take the mean of 50 cells (i,j-50:j)
            # along the y-axis
            thres_d = np.nanmean(step.cmc[0][i[idx[0]], values-50:values])

            # if the concentration mean is larger than 99% in average and that
            # it is not close the axis symmetry, append index
            if (thres_d > 0.99) and (i[idx[0]] > 50):

                jk = np.append(jk, values)
                ik = np.append(ik, i[idx[0]])

        # if empty, the height and rim-to-rim crater diameter is equal to 0
        if (ik.size == 0) or (jk.size == 0):
            alt = np.nan
            drim = np.nan
            iif = np.nan
            ijf = np.nan

        # otherwise, we take the maximum y-index (with the biggest height)
        else:
            idx_depth = np.where(jk == np.max(jk))[0]
            n = np.shape(idx_depth)[0]
            ix_depth = idx_depth[0]

            # if the maximum height is reached in one cell, we just take the value
            if n < 2:
                iif = ik[ix_depth].astype(int)
                ijf = jk[ix_depth].astype(int)
            # if the maximum height is reached in several cells, we take the mean
            else:
                iif = np.int(np.mean(ik[idx_depth]))
                ijf = jk[ix_depth].astype(int)

            # calculate the rim-to-rim crater diameter and rim height
            alt = np.abs((model.y[iif, ijf+1]) -
                         ((1 - step.cmc[0][iif, ijf]) * model.dy))
            drim = (np.abs(model.xc[iif, ijf])) * 2.

    # return also the indexes
    return alt, drim, iif, ijf


'''
***********************************************************************
'''


def craterdimensions(model, mode, id_mat):
    '''
    description:
    calculate all final crater dimensions for 1-layer targets

    inputs:
    model: model output from psp
    mode: mostly equal to 0 (if artifacts along the boundary, can be set to 1)
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)

    outputs:
    time, depth, mdepth, diam, vol, ec, ls, pf, alt, drim, Vrim
    time: crater evolution (time)
    depth: apparent crater depth along the axis of symmetry
    mdepth: maximum apparent crater depth
    diam: apparent crater diameter
    vol: apparent crater volume (below the pre-impact surface)
    ec, ls, pf: number of empty cells, partially filled .....
    alt: height of the rim crest
    drim: rim-to-rim crater diameter
    Vrim: crater volume below the rim crest
    '''

    # calculate the pre-impact surface
    oridy = np.where(model.yy == 0.)[0][0]

    # define all crater variables
    vol = np.zeros(model.nsteps)
    depth = np.zeros(model.nsteps)
    mdepth = np.zeros(model.nsteps)
    diam = np.zeros(model.nsteps)
    time = np.zeros(model.nsteps)
    ec = np.zeros(model.nsteps)
    pf = np.zeros(model.nsteps)
    ls = np.zeros(model.nsteps)
    alt = np.zeros(model.nsteps)
    drim = np.zeros(model.nsteps)
    Vrim = np.zeros(model.nsteps)

    # loop through all saved timesteps
    for u in range(model.nsteps):

        # each timestep are saved in the variable step
        den = model.readStep('Den', u)
        step = copy.deepcopy(den)

        # time in crater evolution
        time[u] = step.time

        # if t = 0
        if time[u] == 0:
            vol[u] = 0
            depth[u] = 0
            mdepth[u] = 0
            diam[u] = 0
            ec[u] = 0
            pf[u] = 0
            ls[u] = 0
            alt[u] = 0
            drim[u] = 0
            Vrim[u] = 0
        else:
            # calculate the x-index (apparent crater diameter)
            ix = ixdiam(step, 0, oridy)

            # if the function ixdiam fail to get a value, then all variables
            # are set to np.nan
            if ((ix == 0) and (time[u] != 0)):
                vol[u] = np.nan
                depth[u] = np.nan
                mdepth[u] = np.nan
                diam[u] = np.nan
                ec[u] = np.nan
                pf[u] = np.nan
                ls[u] = np.nan
                alt[u] = np.nan
                drim[u] = np.nan
            # otherwise we go through all the calculations
            else:
                diam[u] = appdiameter(model, step, time[u], oridy, ix)
                vol[u] = appvolume(model, step, time[u],
                                   id_mat, oridy, diam[u], ix)

                # if mode = 0, we calculate the apparent depth in two different ways
                if mode == 0:
                    depth[u] = midappdepth(
                        model, step, time[u], id_mat, diam[u])
                    mdepth[u] = maxappdepth(
                        model, step, time[u], id_mat, oridy, diam[u], ix)
                # if mode = 1, we calculate only the maximum apparent depth
                elif mode == 1:
                    mdepth[u] = maxappdepth(
                        model, step, time[u], id_mat, oridy, diam[u], ix)

                # last functions
                alt[u], drim[u], ixxx, iyyy = rimrim(
                    model, step, time[u], id_mat, oridy, diam[u], ix)
                Vrim[u] = rimvolume(model, step, time[u],
                                    id_mat, diam[u], ixxx, iyyy)
                ec[u], ls[u], pf[u] = emptycells(
                    model, step, time[u], oridy, diam[u], ix)

    # all variables are returned
    return time, depth, mdepth, diam, vol, ec, ls, pf, alt, drim, Vrim


'''
***********************************************************************
'''


def main(path, folders, paths, mode, id_mat, transient, idx_manual = 0):
    '''
    description:
    calculate all final crater dimensions for 1-layer targets

    inputs:
    path: path containing folders with models (jdata in each modelled case)
    folders: all models
    paths: saving directory
    mode: mostly equal to 0 (if artifacts along the boundary, can be set to 1)
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)
    transient: 0 (by volume)
             : 1 (by the maximum depth along the crater cavity)
    outputs:
    1 .txt file with all crater dimensions every saved timesteps
    1. txt file with all transient crater dimensions
    1. txt file with all final crater dimensions

    each of these text files are saved in different folders
    so 3 txt files are created per models
    '''

    # Change path to working directory
    os.chdir(path)

    # loop through models in folders
    for modelname in folders:

        # change directory to
        patht = path + modelname
        os.chdir(patht)

        # print modelname
        print (modelname)

        # open model with pySALEplot
        model = psp.opendatfile('jdata.dat')

        # calculate all final crater dimensions
        time, depth, mdepth, diam, vol, ec, ls, pf, alt, drim, Vrim = craterdimensions(
            model, mode, id_mat)

        ####################################################
        # CALCULATE CRATER DIMENSIONS DURING CRATER EVOLUTION (EACH STEP)
        ####################################################

        # create plots directory if not existing
        path_data = paths + modelname + "/evolution/"

        if not os.path.exists(path_data):
            os.makedirs(path_data)

        os.chdir(path_data)

        # export data to .txt file
        header_txt = "time\tdepth\tmdepth\tdiameter\tvolume\tls\tec\tpf\talt\tdrim\tVrim"

        # we need to define the name of the txt file that will be saved
        fname = modelname + '_data.txt'

        # output file
        output = np.column_stack(
            (time, depth, mdepth, diam, vol, ls, ec, pf, alt, drim, Vrim))
        np.savetxt(fname, output, header=header_txt,
                   delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])

        ####################################################
        # CALCULATE CRATER DIMENSIONS AT THE TRANSIENT CRATER
        ####################################################

        # create plots directory if not existing
        path_data = paths + modelname + "/transient/"

        if not os.path.exists(path_data):
            os.makedirs(path_data)

        os.chdir(path_data)

        fname = modelname + '_tr.txt'
        header_txt = "tr_time\ttr_depth\ttr_diameter\ttr_vol\talt\td_rim\tV_rim\tidx"

        # the transient crater diameter is defined at the maximum volume
        if transient == 0:
            idx_tr = np.nanargmax(vol)
        else:
            idx_tr = np.nanargmax(mdepth)  # if maximum depth is used
        # transient crater values are stored in another .txt file and folder
        if mode == 0:
            output2 = np.column_stack(
                (time[idx_tr], depth[idx_tr], diam[idx_tr], vol[idx_tr], alt[idx_tr], drim[idx_tr], Vrim[idx_tr], idx_tr))
            np.savetxt(fname, output2, header=header_txt,
                       delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
        else:
            output2 = np.column_stack(
                (time[idx_tr], mdepth[idx_tr], diam[idx_tr], vol[idx_tr], alt[idx_tr], drim[idx_tr], Vrim[idx_tr], idx_tr))
            np.savetxt(fname, output2, header=header_txt,
                       delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])

        ####################################################
        # CALCULATE CRATER DIMENSIONS AT THE FINAL CRATER
        ####################################################

        # it should take the value at t/tr = 10 and not as the mean of the last 5 values, it could make some of the other codes easier
        # the problem that I should not take any values maybe I should say that it should be at least

        # create plots directory if not existing
        path_data = paths + modelname + "/final/"

        if not os.path.exists(path_data):
            os.makedirs(path_data)

        os.chdir(path_data)

        # at t/tr for different times
        tnorm = time / time[idx_tr]
        
        # save for t/tr = 2, 3, 4, 5, 7.5 and 10.0        
        tnorm_fname = ['020', '030', '040', '050', '070', '100']
        
        for iv, tnormv in enumerate([2.0, 3.0, 4.0, 5.0, 7.0, 10.0]):
            
            __, stp1 = find_nearest(tnorm, tnormv)
    
            fname = modelname + '_final' + tnorm_fname[iv] + '.txt'
            header_txt = "f_time\tf_depth\tf_diameter\tf_vol\tf_alt\tf_drim\tf_Vrim"
    
            # if it did not reach the final crater, then nothing should be save in the final folder
            if tnorm[np.where(~np.isnan(diam))[0][-1]] >= tnormv - 0.20:
    
                # final crater values are stored in another .txt file and folder
                if mode == 0:
                    output2 = np.column_stack((np.nanmean(time[stp1]), np.nanmean(depth[stp1]), np.nanmean(
                        diam[stp1]), np.nanmean(vol[stp1]), np.nanmean(alt[stp1]), np.nanmean(drim[stp1]), np.nanmean(Vrim[stp1])))
                    np.savetxt(fname, output2, header=header_txt,
                               delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
                else:
                    output2 = np.column_stack((np.nanmean(time[stp1]), np.nanmean(mdepth[stp1]), np.nanmean(
                        diam[stp1]), np.nanmean(vol[stp1]), np.nanmean(alt[stp1]), np.nanmean(drim[stp1]), np.nanmean(Vrim[stp1])))
                    np.savetxt(fname, output2, header=header_txt,
                               delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
    
            else:
                if mode == 0:
                    output2 = np.column_stack(
                        (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                    np.savetxt(fname, output2, header=header_txt,
                               delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
                else:
                    output2 = np.column_stack(
                        (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                    np.savetxt(fname, output2, header=header_txt,
                               delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
                
                
        ####################################################################
        # CALCULATE CRATER DIMENSIONS AT THE FINAL CRATER (MANUAL)
        ####################################################################

        # it should take the value at t/tr = 10 and not as the mean of the last 5 values, it could make some of the other codes easier
        # the problem that I should not take any values maybe I should say that it should be at least

        # create plots directory if not existing
        path_data = paths + modelname + "/manual/"

        if not os.path.exists(path_data):
            os.makedirs(path_data)

        os.chdir(path_data)

        # only run the script if idx_manual is specified
        if idx_manual > 0:
            stp1 = idx_manual
    
            fname = modelname + '_final.txt'
            header_txt = "f_time\tf_depth\tf_diameter\tf_vol\tf_alt\tf_drim\tf_Vrim"
            
            if mode == 0:
                output2 = np.column_stack((np.nanmean(time[stp1]), np.nanmean(depth[stp1]), np.nanmean(
                                           diam[stp1]), np.nanmean(vol[stp1]), np.nanmean(alt[stp1]), np.nanmean(drim[stp1]), np.nanmean(Vrim[stp1])))
                np.savetxt(fname, output2, header=header_txt,
                                           delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
            else:
                output2 = np.column_stack((np.nanmean(time[stp1]), np.nanmean(mdepth[stp1]), np.nanmean(
                                           diam[stp1]), np.nanmean(vol[stp1]), np.nanmean(alt[stp1]), np.nanmean(drim[stp1]), np.nanmean(Vrim[stp1])))
                np.savetxt(fname, output2, header=header_txt,
                                           delimiter='\t', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e', '%1.6e'])
                
        else:
            None

        # close model file
        model.closeFile()


'''
***********************************************************************
'''


def rimrimProfile(model, step, drim, id_mat):
    '''
    description:
    return the profile of the crater up to the rim-to-rim crater diameter
    at the chosen saved timestep, but return the saved profile as 11 points,
    as in Watters et al. 2015

    inputs:
    model: model output from psp
    step: saved timestep loaded from psp
    drim: rim-to-rim crater diameter
    id_mat: material id (0: all, 1: projectile, 2: first layer, 3: second ...)

    outputs:
    X, Y
    '''

    # calculate the maximum distance + 10%
    maxdist = (drim/2.) * 1.1

    # get the x-index
    iiix = np.int(np.ceil(maxdist / model.dy))

    # select the concentration in cells for a specific array
    conc2 = step.cmc[id_mat][:iiix, :]

    # save x- and y-index
    ii = np.arange(iiix)
    jj = np.array([])

    for values in ii:

        # detect boundary using an average value over 20 cells. If there is less
        # than 1% in average over 20 cells, then the boundary is detected
        # j-index can be appended
        ma = movingaverage(conc2[values, :], 20)
        tijf = np.where(ma < 0.01)[0][0]  # not the best way to constrain it
        jj = np.append(jj, tijf)

    jj = jj.astype(int)
    altr = np.array([])
    rrim = np.array([])

    # for each x- and y-index the distance and height is calculated
    for values in np.arange(len(jj)):

        a1 = (model.y[ii[values], jj[values]+1]) - \
            ((1 - step.cmc[0][ii[values], jj[values]]) * model.dy)
        a2 = (np.abs(model.xc[ii[values], jj[values]]))
        altr = np.append(altr, a1)
        rrim = np.append(rrim, a2)

    # if we want to have divide it as in Watters et al. 2015 (in eleven points,
    # or letters in his case)
    rj = (drim/2.) * 1.1

    rlinspace = np.linspace(0, rj, 11)

    # create new empty array
    altg = np.array([])
    rg = np.array([])

    # find the nearest values
    for values in rlinspace:

        rr, idx = find_nearest(rrim, values)

        ll = altr[idx]
        altg = np.append(altg, ll)
        rg = np.append(rg, rr)

    # return data
    return rg, altg


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

def power(x,a,b,c):
    
    return a * (x**b) + c
    

    
def CAVITY_SHAPE_EXPONENT(r,d):
    
    #Cavity shape exponent from B to E
    
    a, b = curve_fit(power,r[1:9],d[1:9])
        
    exponent = a[1]
    
    return exponent    

***********************************************************************
'''

def get_cross_section(model, step, id_mat, oridy):
    
    # find the maximum diameter along the pre-impact surface
        ixx = ixdiam(step, id_mat, oridy)

        # get the concentration
        extra = np.int(np.round(((model.nx - ixx) / 1.5) + ixx))
        conc = step.cmc[0][:extra, :]  # + 1 before

        # define the x- and y-indexes
        ii = np.arange(extra)
        jj = np.array([])

        # values are append to y-index if.....
        for values in ii:

            ma = movingaverage(conc[values, :], 20)
            lll = np.where(ma < 0.01)[0]
            if lll.size == 0:
                tijf = np.where(ma == np.min(ma))[0][0]
            else:
                tijf = lll[0]
            jj = np.append(jj, tijf)

        # create empty arrays
        jj = jj.astype(int)
        altr = np.array([])
        rrim = np.array([])

        # calculate the height and distance
        for values in np.arange(len(jj)):

            a1 = (model.y[ii[values], jj[values]+1]) - \
                ((1 - step.cmc[0][ii[values], jj[values]]) * model.dy)
            a2 = (np.abs(model.xc[ii[values], jj[values]]))
            altr = np.append(altr, a1)
            rrim = np.append(rrim, a2)
            
            
        return (rrim, altr)
'''
***********************************************************************
'''


def craterProfiles(folders, path, paths, idx, mode):
    '''
    description:
    return the profile of the crater for the transient crater (mode = 1), 
    the final crater (mode = 2), or for a chosen time step (mode = 3, and idx = stp)    

    input:
    mode = 1 # transient crater
    mode = 2 # final crater
    mode = 3 # any other craters

    if mode = 1 or 2, idx does not count
    if mode = 3, the values in idx will be taken

    example:
    folders = ['C00P20F06_L250','C00P20F06_L500']
    path = '/media/nilscp/Cloud1/SENS/PORFRIC/results/'
    paths = '/media/nilscp/Zell/Collapse/plots2/'

    #transient
    idx = 0
    mode = 1
    craterProfiles(folders, path, paths, idx, mode)

    #final
    idx = 0
    mode = 2
    craterProfiles(folders, path, paths, idx, mode)

    #arbitrary (timestep = 4)
    idx = 4
    mode = 3
    craterProfiles(folders, path, paths, idx, mode)
    '''

    # we can loop through models
    for ix, modelname in enumerate(folders):

        # change to working directory
        patht = paths + modelname
        os.chdir(patht)

        # print modelname
        print (modelname)

        # id_mat
        id_mat = 0

        # load jdata.data
        model = psp.opendatfile(path + modelname + '/jdata.dat')

        # Get the y-index of the pre-surface
        oridy = np.where(model.yy == 0.)[0][0]
        
        # get the time step (in sec)
        den1 = model.readStep('Den', 1)
        timestep = den1.time
        
        # get the total time of the simulation
        total_time = model.laststep * timestep
        
        # get the index at when the transient crater is reached
        tr_time , __, __, __, __, __, __, idx_tr = np.loadtxt(paths + modelname + 
                                                        '/transient/' + 
                                                        modelname + '_tr.txt')
        
        idx_tr = idx_tr.astype('int')

        # get the total time of the simulation (in normalized time)
        total_time_norm = total_time / tr_time
        
        # default normalized time
        default_ntime = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
        default_ntime = default_ntime.astype('int')
        default_nfname = np.array(['020', '030', '040', '050', '070', '100'])
        
        # where we have actually data for
        real_ntime = default_ntime[default_ntime <= total_time_norm]
        real_nfname = default_nfname[default_ntime <= total_time_norm]
        
        # load data depending on the mode (if transient only)
        if mode == 1:
            
            ####################################################
            # TRANSIENT CRATER
            ####################################################            
            
            den = model.readStep('Den', idx_tr)
            step = copy.deepcopy(den)
            (rrim, altr) = get_cross_section(model, step, id_mat, oridy)
            
            path_to_save = paths + modelname + '/transient/'

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            os.chdir(path_to_save)
            
            fname = modelname + '_XYtransientprofile.txt'
            
            header_txt = "distance\tdepth"

            output = np.column_stack((rrim, altr))

            np.savetxt(fname, output, header=header_txt,
                       delimiter='\t', fmt=['%1.6e', '%1.6e'])
            
        elif mode == 3:
            
            ####################################################
            # SELECTED CRATER
            ####################################################               
            
            den = model.readStep('Den', idx)
            step = copy.deepcopy(den)
            (rrim, altr) = get_cross_section(model, step, id_mat, oridy)
            
            path_to_save = paths + modelname + '/arbitrary/'

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            os.chdir(path_to_save)
            
            fname = modelname + '_XY' + str(idx) + '_profile.txt'
            
            header_txt = "distance\tdepth"

            output = np.column_stack((rrim, altr))

            np.savetxt(fname, output, header=header_txt,
                       delimiter='\t', fmt=['%1.6e', '%1.6e'])
            
        else: # mode == 2

            ####################################################
            # FINAL CRATER at different normalized time
            ####################################################  
        
            for ir, r in enumerate(real_ntime):
                
                idx = r * idx_tr
                den = model.readStep('Den', idx)
                step = copy.deepcopy(den)
                (rrim, altr) = get_cross_section(model, step, id_mat, oridy)
                
                path_to_save = paths + modelname + '/final/'

                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
    
                os.chdir(path_to_save)
                fname = modelname + '_XYfinalprofile' + real_nfname[ir] + '.txt'
                
                header_txt = "distance\tdepth"

                output = np.column_stack((rrim, altr))

                np.savetxt(fname, output, header=header_txt,
                           delimiter='\t', fmt=['%1.6e', '%1.6e'])

        # close model file
        model.closeFile()

    print ("transient/final crater profiles are saved")


'''
***********************************************************************
'''


def craterProfileXY(model, stp, id_mat, layerdepth):
    '''
    description:
    return the profile of the crater for a chosen time step, works also
    with different materials and multi-layered targets

    inputs:
    model: jdata of modelled case
    stp: saved timestep
    id_mat: material(s) id(s)
    layerdepth: depth of the layer (upper layer, layerdepth = 0)
    '''

    # Get the y-index of the pre-surface
    oridy = np.where(model.yy == layerdepth)[0][0]

    # load data depending on the mode
    den = model.readStep('Den', stp)

    # make a hard copy
    step = copy.deepcopy(den)

    # find the maximum diameter along the pre-impact surface
    ixx = ixdiam(step, id_mat, oridy)

    # get the concentration
    extra = np.int(np.round(((model.nx - ixx) / 1.5) + ixx))
    conc = step.cmc[id_mat][:extra, :]  # + 1 before

    ii = np.arange(extra)  # ix before
    jj = np.array([])

    for values in ii:

        ma = movingaverage(conc[values, :], 20)
        lll = np.where(ma < 0.01)[0]
        if lll.size == 0:
            tijf = np.where(ma == np.min(ma))[0][0]
        else:
            tijf = lll[0]
        jj = np.append(jj, tijf)

    jj = jj.astype(int)
    altr = np.array([])
    rrim = np.array([])

    for values in np.arange(len(jj)):

        a1 = (model.y[ii[values], jj[values]+1]) - \
            ((1 - step.cmc[id_mat][ii[values], jj[values]]) * model.dy)
        a2 = (np.abs(model.xc[ii[values], jj[values]]))
        altr = np.append(altr, a1)
        rrim = np.append(rrim, a2)

    # close model file
    model.closeFile()

    return (rrim, altr)


'''
***********************************************************************
'''


import re


def tokenize(filename):
    '''
    Function to list filenames in correct order (see
    http://stackoverflow.com/questions/5997006/sort-a-list-of-files-using-python)

    :param filename:
    :return:
    '''
    digits = re.compile(r'(\d+)')

    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))


'''
***********************************************************************
'''


def morph2layers(model, t):
    '''
    description:
    calculate both the crater dimensions and crater profiles of two-layer targets

    input: 
    model: jdata
    t: thickness of the upper layer (here we use a positive value)
    '''

    # get the index-x upper layer
    oridy = np.where(model.yy == 0.)[0][0]

    # get the closest index-x lower layer (in case where it is not exactly equal)
    if np.size(np.where(model.yy == -t)[0]) == 0:
        oridy2 = np.argmin(np.abs(model.yy+t))
    else:

        oridy2 = np.where(model.yy == -t)[0][0]

    # load last step
    # density seems to be better : ) e.g with C20P00F08_L100
    den = model.readStep('Den', model.nsteps-1)
    step = copy.deepcopy(den)

    # save time
    time = step.time

    # Calculations for the upper layer
    ix = ixdiam(step, 0, oridy)  # ix to all material
    DAPP_UL = appdiameter(model, step, time, oridy, ix)  # Dapp to all material
    MD_UL = maxappdepth(model, step, time, 0, oridy,
                        ix)  # dapp to all material
    HRIM_UL, DRIM_UL = rimrim(model, step, time, 0, oridy, ix)  # DRIM

    # Calculations for the lower layer (the apparent crater diameter is
    # calculated in two different ways)
    ix2 = ixdiam(step, 2, oridy2)  # id material 2 (only lower layer)
    ix3 = ixdiam(step, 0, oridy2)  # id material 0 (all layers)

    DAPP_LL = appdiameter(model, step, time, oridy2, ix3)  # upper material
    DAPP_LL2 = appdiameter(model, step, time, oridy2, ix2)  # upper material
    MD_LL = maxappdepth(model, step, time, 2, oridy,
                        ix3)  # dapp to second material
    # it gives a positive depth (but it should be negative) DRIM mat 2 (it does not manage to get the rim)
    HRIM_LL, DRIM_LL = rimrim(model, step, time, 2, oridy2, ix3)

    # cross section profiles
    X, Y = craterProfileXY(model, step, 0, 0.)
    X2, Y2 = craterProfileXY(model, step, 2, -t)

    ''' 
    import matplotlib.pyplot as plt
    path = '/var/tmp/mysshfs/stallo/layering/vdisc/results/C5KC50K_L200/'
    os.chdir(path)
    model = psp.opendatfile('jdata.dat')
    t=400 # meters

    (DAPP_UL,MD_UL,HRIM_UL,DRIM_UL,
     DAPP_LL,MD_LL,HRIM_LL,DRIM_LL,DAPP_LL2,
     XUL,YUL,XLL,YLL) = morph2layers(model,t)
          
     plt.plot(XUL,YUL,'bo')
     plt.plot(XLL,YLL,'ro')
     plt.plot(DAPP_UL/2.,0,'ko',ms=15)
     plt.plot(DAPP_LL/2.,-t,'ko',ms=15)
     plt.plot(DAPP_LL2/2.,-t,'ko',ms=15)
     plt.plot(DRIM_LL/2.,-HRIM_LL,'ko',ms=15)
     plt.plot(DRIM_UL/2.,HRIM_UL,'ko',ms=15)

     # In order to loop
    
    path = "/media/nilscp/Yuna/VDISC/results/"
    # Change path to working directory
    os.chdir(path)
    
    # select folders in directory
    folders = glob.glob("C005K*")
    folders.sort(key=tokenize)
    L = np.arange(100,460,10)
    L = np.append(L,(500,600,700,800,900,1000))
    
    t=400 # meters
    
    DAPP_UL = np.zeros(len(folders))
    MD_UL = np.zeros(len(folders))
    HRIM_UL = np.zeros(len(folders))
    DRIM_UL = np.zeros(len(folders))
    DAPP_LL = np.zeros(len(folders))
    MD_LL = np.zeros(len(folders))
    HRIM_LL = np.zeros(len(folders))
    DRIM_LL = np.zeros(len(folders))
    DAPP_LL2 = np.zeros(len(folders))
    
    for ix, modelname in enumerate(folders):
        
        patht = path + modelname
        os.chdir(patht)
        
        # print modelname
        print modelname
       
    
        model = psp.opendatfile('jdata.dat')
        

        (DAPP_UL[ix],MD_UL[ix],HRIM_UL[ix],DRIM_UL[ix],
         DAPP_LL[ix],MD_LL[ix],HRIM_LL[ix],DRIM_LL[ix],DAPP_LL2[ix],
         XUL,YUL,XLL,YLL) = morph2layers(model,t)
         
         # change to saving path
        paths = "/media/nilscp/Yuna/VDISC/plots/"
        path_to_save = paths + modelname + '/data/'
        
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        os.chdir(path_to_save)
        
        
        # DATA PROFILE
        fname = modelname + '_finalprofileUL.txt'
        header_txt = "distance\tdepth"
        
        # index where the transient crater is reached  
        output = np.column_stack((XUL, YUL)) 
        
        np.savetxt(fname, output, header = header_txt ,
                       delimiter='\t', fmt=['%1.6e','%1.6e'])
                       
        fname2 = modelname + '_finalprofileLL.txt'
        header_txt = "distance\tdepth"
        
        # index where the transient crater is reached  
        output2 = np.column_stack((XLL, YLL)) 
        
        np.savetxt(fname2, output2, header = header_txt ,
                       delimiter='\t', fmt=['%1.6e','%1.6e'])
         
        model.closeFile()
    
    import matplotlib.pyplot as plt    
    plt.plot(L,DAPP_UL,"ko")
    plt.plot(L,DAPP_LL,"ro")
    plt.plot(L,DAPP_LL2,"yo")
    
    plt.plot(L,DRIM_UL,"ko")
    plt.plot(L,DRIM_LL,"ro")
    '''

    return (DAPP_UL, MD_UL, HRIM_UL, DRIM_UL,
            DAPP_LL, MD_LL, HRIM_LL, DRIM_LL, DAPP_LL2,
            X, Y, X2, Y2)


'''
***********************************************************************
'''
