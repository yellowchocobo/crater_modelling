# -*- coding: utf-8 -*-

'''
******************************************************************************

     =========================================================================
     Subroutine to read, post-process and save crater dimensions. Call subroutines
     such as ejecta.py and crater.py

     Description                                     Programmer    Date
     ------------------------------------------------------------------
     Original version (1.0).............................NCP  2016/01/12
     Modified version (2.0).............................NCP  2017/11/23
     xxx                      ..........................NCP  xxxx/xx/xx
     xxx                      ..........................NCP  xxxx/xx/xx
    
    The version 2.0 includes:
    - a better description of functions
    - changed the name of some function 
    - functions:
    
    
    ==========================================================================

******************************************************************************
'''

# import Python's module
import sys
import glob
import os

# add directories, which contain my in-house scripts
#pathm = ['/work/nilscp/Python/prog/export', '/work/nilscp/Python/prog/clean', '/work/nilscp/Python/prog/import',
#         '/work/nilscp/Python/prog/scal', '/work/nilscp/Python/Papers/Paper3']


pathm = ['/uio/kant/geo-ceed-u1/nilscp/Nils/Python/craterGeomorph']


# add directories to system path
for pat in pathm:
    sys.path.append(pat)

# import various in-house sub-routines
#import export
#import ejecta as ej
import crater as cr
#import plotting
#import read as readd
'''
***********************************************************************
'''

###############################################################################
# 0. INPUT
###############################################################################

# INPUT (1)
# or 3 dots to calculate the velocity (does not influence anything here)
method = 3

# INPUT (2, 3 and 4)
# related to how the depth of the crater is calculated (1 is a good value)
mode = 1
poros = 1  # if porous = 1, non porous = 0
id_mat = 0  # 0 = everything, 2 only lower material, especially practical in layering modelling
transient = 0  # 0 transient at max volume, 1: transient at maximum depth
thresholdf = 0.01
g = 1.62

'''
***********************************************************************
'''


# main directory which contains models with jdata
#path = '/var/tmp/mysshfs/stallo/collapse/ART3/results/'
pathm = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/benchmarkFI/CPPR/UAVG/CPPR5/'

folders =  ['CDILNOA', 'CDILNOP', 'CDILWPA', 'CDILWPA2', 'CDILWPO', 'CDILWPO2', 
            'COLDNOP', 'COLDWPO', 'COLDWPO2', 'IVANOVS', 'SANDCOH']


for f in folders:
    
    path = pathm + f + '/'

    os.chdir(path)

    # selection of folders
    search = '*'
    models = glob.glob(search)

    pathdata = '/run/media/nilscp/Squall/benchmarkFI/CPPR5/data/'
    pathplots = '/run/media/nilscp/Squall/benchmarkFI/CPPR5/plots/'

###############################################################################
# 1. CALCULATE EXCAVATED CRATER DIMENSIONS
###############################################################################

# Ve, de, De and shape of cavities
#ej.main(path, folders, pathdata, method, thresholdf, g)


###############################################################################
# 2. CALCULATE CRATER DIMENSIONS DURING CRATER EVOLUTION
# 3. CALCULATE CRATER DIMENSIONS AT THE TRANSIENT CRATER
# 4. CALCULATE CRATER DIMENSIONS AT THE FINAL CRATER (LAST FIVE TIMESTEPS)
###############################################################################

    cr.main(path, models, pathdata, mode, id_mat, transient)


###############################################################################
# 5. SAVE TXT FILE WITH X AND Y FOR THE TRANSIENT and FINAL CRATER
###############################################################################
    cr.craterProfiles(models, path, pathdata, 0, 1)  # XY transient craters
    cr.craterProfiles(models, path, pathdata, 0, 2)  # XY final craters
# evo.XY_PROFILE(folders, path, pathdata, 19, 3) # arbitrary


###############################################################################
# 6. MAKE A FEW PLOTS IN ORDER TO HAVE THE POSSIBILITY TO DOUBLE CHECK SCRIPTS
# better to save plots in the same folder for all of them
###############################################################################
plotting.main(pathdata, folders, pathplots)

###############################################################################
# 7. NOTE IN GOOGLE DRIVE THAT THE RESULTS ARE FREE OF ERRORS
###############################################################################

'''
***********************************************************************
'''

###################################################
# 2. EXTRACT INFORMATION FROM MODELS, READ PARAMETER
# AND COMPUTE SCALING LAWS (PI2, PIV, PID)
###################################################
nlayers = 1
paths = pathdata
for idx, modelname in enumerate(folders):
    pathi = path + modelname + '/INFO'
    readd.param(pathi, poros, paths + modelname +
                '/data/', nlayers)  # generate python_param
    # export.scalinglawsDRIM(paths + modelname + '/data/', modelname, nlayers) # generate scalinglaws DRIM
    export.scalinglaws(paths + modelname + '/transient/',
                       modelname, nlayers)  # generate scalinglaws


name = 'HOM_AUG'
search = 'C*'
# generate a table with L, pi2, pi3, piD ....
export.tablepi(paths, search, name, nlayers)
export.tablepiDRIM(paths, search, name, nlayers)
'''
***********************************************************************
'''
###################################################
# 3. READ PREVIOUSLY CREATED FILES
# AND COMPUTE SCALING LAWS
###################################################


filename = name + '_table.csv'
data = readd.pandas(filename)


'''
***********************************************************************
'''

###################################################
# 4. COMPUTE SCALING LAWS factor (beta, mu and K2)
# from 2.
###################################################

os.chdir(paths)

sea = 'C*'
lst = export.selecttargprop(paths, sea)

# Lmin = 50 # minimum prjectile length to take into account - important when models are not completely finished
#Lmax = 5000
Dsc = 16000
name = 'Call'

export.piDpiV(paths, lst)  # save pi2, piD in piD folder

# THIS IS ONLY FOR cohesionless materials (or materials with low cohesion)
# take previously generated file and calculate factors
export.savefactors(paths + 'piDpiV/', lst, name, Dsc)

'''
***********************************************************************
'''

###################################################
# 5. PLOT SCALING LAWS
###################################################

os.chdir(paths + 'piDpiV/')
lst = glob.glob('C*')
plott.piD(path, lst)  # save piD (first inspection check)

# rocks = data.loc[(output["por"]== 0.12) & (output["Friction_int"] == 1.0) & (output["Cohesion_int"] == 1e6)]

'''
***********************************************************************
'''

###################################################
# 6. GET TRANSIENT CRATERS
###################################################

os.chdir(path)
folders = glob.glob(search)  # or folders = ['CP20F']

trt = export.GET_TRANSIENT_TIME(folders, paths)
morph.SELECT_PROFILE(folders, trt, path, paths)

# plot transient craters
plott.transientPROFILE(paths, folders)


###################################################
# 6. GET RIM-RIM CRATERS
###################################################


path = '/var/tmp/mysshfs/stallo/layering/vdisc/resultsHOM/'
os.chdir(path)
folders = glob.glob('C*')
paths = '/media/nilscp/Ward/layering/vdisc/plots/P10/scalinglaws2/'
morph.SELECT_PROFILElast(folders, path, paths)
#SELECT_PROFILElast(folders, path, paths)
