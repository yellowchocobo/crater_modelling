#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:24:08 2018

@author: nilscp
"""

import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

path_pySALEPlot = '/work/nilscp/iSALE/Dellen/lib'

# path to pySALEPlot is loaded
sys.path.append(path_pySALEPlot)
import pySALEPlot as psp

# add directories, which contain my in-house scripts
pathm = ['/work/nilscp/Python/prog/export', '/work/nilscp/Python/prog/clean', '/work/nilscp/Python/prog/import',
         '/work/nilscp/Python/prog/scal', '/work/nilscp/Python/Papers/Paper3']

# add directories to system path
for pat in pathm:
    sys.path.append(pat)

# import various in-house sub-routines
#import export
import ejecta as ej


'''
***********************************************************************
'''
path = '/work/nilscp/iSALE/isaleruns/testruns/ejecta/L20km_part1_HYDRO_CPPR1000/results/L20km_HYDRO_part2_CPPR80g/'
path = '/work/nilscp/iSALE/isaleruns/testruns/ejecta/L20km_part1_HYDRO_CPPR1000/results/L20km_HYDRO_part1g/'
path = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/ejecta/results/length/a2km/'
path = '/uio/kant/geo-ceed-u1/nilscp/Desktop/stallo_work/ejecta/results/length/a2km_C1000/'

path = '/work/nilscp/iSALE/isaleruns/testruns/multilayers/NILS_L100/'

'''
***********************************************************************
'''

os.chdir(path)
model = psp.opendatfile('jdata.dat')

'''
***********************************************************************
'''
# print a single saved timesteps
u = 0

den = model.readStep(['Den', 'Yld'], u)
step = copy.deepcopy(den)

a = 2000.
plt.pcolormesh(model.x/a, model.y/a, step.cmc[0], cmap='Pastel1')

# plot the high resolution zone
plt.hlines(model.yhires[0]/a, model.xhires[0]/a,
           model.xhires[1]/a, 'k', linewidth=2)
plt.hlines(model.yhires[1]/a, model.xhires[0]/a,
           model.xhires[1]/a, 'k', linewidth=2)
plt.vlines(0, model.yhires[0]/a, model.yhires[1]/a, 'k', linewidth=2)
plt.vlines(model.xhires[1]/a, model.yhires[0]/a,
           model.yhires[1]/a, 'k', linewidth=2)

plt.hlines(1000./1000., model.xhires[0]/a,
           model.xhires[1]/a, 'r', linewidth=4)  # Z = 0.1 R

'''
***********************************************************************
'''

# or calculated the ejecta data
tsa = (model.objrad[0]*2.) / (12700.)
method = 3
g = 1.62
threshold = (model.cppr[0]*model.dx) * 0.01
X, Y, ix, ts, dt, n = ej.posTracer(model, method, threshold)
ix = ix.astype('int')

# or
nix = np.where(ts >= 2)
v, angle, xpos = ej.parameters(X[nix], Y[nix], nix, model, method, dt)
tair, dland = ej.ballistic(v, angle, xpos, nix, g)

# or
v, angle, xpos = ej.parameters(X, Y, ix, model, method, dt)

# because of the tracers script the last step is not saved
xtime = np.linspace(0, dt*199., 199.)

# because of the tracers script the last step is not saved
xtime = np.linspace(0, dt*99., 99.)

###
g = 1.62
tair, dland = ej.ballistic(v, angle, xpos, ix, g)


'''
***********************************************************************

import glob
import numpy as np

patht = '/work/nilscp/iSALE/isaleruns/testruns/ejecta/L20km_part1_HYDRO_CPPR1000/results/'
os.chdir(patht)

# selection of folders
search = 'L20km_HYDRO_part2*'
folders = glob.glob(search) 

#folders =  ['C00P20F08_L250', 'C00P20F08_L500','C00P20F08_L1000']
#folders =['ACFDIL_NOPOR_108_L700']

#should maybe only care about the target material and not projectile material


pathdata = '/work/nilscp/iSALE/isaleruns/data/ejecta/'
pathplots = '/work/nilscp/iSALE/isaleruns/data/ejecta/'
###############################################################################
## 0. INPUT
###############################################################################

# INPUT (1)
method = 2 # or 3 dots to calculate the velocity (does not influence anything here)

x, y, tracer_idx, timesteps, dt, n = ej.posTracer(model,method)
v, angle, xpos = ej.parameters(x,y,tracer_idx,model,dt)
tracer_idx  = tracer_idx.astype(int)

'''

# number of newly detected ejected particles per timestep ## FIGURE 1 ##
# plt.plot(np.arange(0,len(n))/2.,n,"ko")
plt.plot(xtime/tsa, (n/3930.)*100., "ko")
plt.xlabel('t_s', fontsize=16)
plt.ylabel('n tracers', fontsize=16)
plt.title('CPPR = 50', fontsize=20)
plt.ylim(0,)
plt.xlim(0, 110)

nCPPR50 = n
nCPPR50norm = (n/3930.)*100.

# export data to .txt file
header_txt = "time;ntime;ntracers;nntracers"

# we need to define the name of the txt file that will be saved
fname = 'tracers_data.txt'

# output file
output = np.column_stack((xtime, xtime/tsa, n, n/3930.))
np.savetxt(fname, output, header=header_txt,
           delimiter=';', fmt=['%1.6e', '%1.6e', '%1.6e', '%1.6e'])

#########################################################################
plt.plot(xtime/tsa, nCPPR20norm, "ko", label='CPPR20')
plt.plot(xtime/tsa, nCPPR40norm, "ro", label='CPPR40')
plt.plot(xtime/tsa, nCPPR80norm, "bo", label='CPPR80')
plt.xlabel('t/t_s', fontsize=16)
plt.ylabel(r'$n_{ej}/n_{proj}$', fontsize=16)
plt.legend()
plt.ylim(0,)
plt.xlim(0, 110)

# xpos vs velocity (with timesteps in angles) ## FIGURE 2 ##
plt.scatter(xpos/2000., v/12700., c=timesteps, cmap='jet',
            vmin=0, vmax=100, s=4, linewidths=0)
plt.colorbar()
# plt.xscale('log')
plt.yscale('log')
plt.xlabel('xpos/R', fontsize=16)
plt.ylabel('Vej/U', fontsize=16)
plt.xlim(0, 2)
plt.ylim(1e-2, 1)
plt.title('CPPR = 50', fontsize=20)
plt.tight_layout()

# xpos vs angle (with timesteps in angles) ## FIGURE 3 ##
plt.scatter(xpos/10000., angle, c=timesteps, cmap='jet',
            vmin=0, vmax=200, s=4, linewidths=0)
plt.colorbar()
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('xpos/R', fontsize=16)
plt.ylabel('angle', fontsize=16)
plt.xlim(0, 5)
plt.ylim(0, 90)
plt.title('CPPR = 1000', fontsize=20)
plt.tight_layout()


# xpos vs angle (with timesteps in angles) ## FIGURE 3 ##
plt.scatter(v/12700., angle, c=timesteps, cmap='jet',
            vmin=0, vmax=200, s=4, linewidths=0)
plt.vlines(2400./12700., 0, 90, 'k')
plt.colorbar()
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Vej/U', fontsize=16)
plt.ylabel('angle', fontsize=16)
plt.xlim(1e-4, 1)
plt.ylim(0, 90)
plt.title('CPPR = 80', fontsize=20)
plt.tight_layout()


'''
***********************************************************************
'''
# FIG 2
import matplotlib
plt.scatter(xpos/2000., angle, c=v/12700., cmap='jet', vmin=1e-3,
            vmax=1, s=4, linewidths=0, norm=matplotlib.colors.LogNorm())
clb = plt.colorbar()
clb.set_label(r'$V_{ej}/U$', fontsize=14, labelpad=-35, y=1.075, rotation=0)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('$x_{pos} / a$', fontsize=16)
plt.ylabel('$ Angle ' + ' (^{o})$', fontsize=16)
plt.xlim(1.0, 20.0)  # plt.xlim(0.5,2.0)
plt.ylim(0, 90)
plt.title('CPPR = 50, Z = 0.01a', fontsize=20)
plt.tight_layout()

'''
***********************************************************************
'''

# FIG3

den = model.readStep('Den', 0)
stepor = copy.deepcopy(den)

Tmp = model.readStep('Tmp', 0)
stepor = copy.deepcopy(Tmp)

denm = model.readStep('Den', 200)
stepend = copy.deepcopy(denm)

# original position
plt.pcolormesh(model.xc/(2000.), model.yc/(2000.), stepor.mat, cmap='Pastel1')
plt.plot(stepor.xmark[ix]/(2000.), stepor.ymark[ix]/(2000.), 'ko')
plt.xlim(0, 15)  # plt.xlim(0,4)
plt.ylim(-4, 4)  # plt.ylim(-2,2)
plt.xlabel('$x / a$', fontsize=16)
plt.ylabel('$y / a$', fontsize=16)
plt.title('CPPR = 50, Z = 0.01a', fontsize=20)
plt.tight_layout()


# FIG 4
import matplotlib
plt.scatter(v/12700., angle, c=dland/2000., cmap='jet', vmin=1,
            vmax=1000, s=4, linewidths=0, norm=matplotlib.colors.LogNorm())
clb = plt.colorbar()
clb.set_label(r'$land_{pos}/a$', fontsize=14,
              labelpad=-35, y=1.075, rotation=0)
plt.vlines(2380./12700., 0, 90)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'$V_{ej}/U$', fontsize=16)
plt.ylabel('$ Angle ' + ' (^{o})$', fontsize=16)
plt.xlim(1e-4, 1)  # plt.xlim(0.5,2.0)
plt.ylim(0, 90)
plt.title('CPPR = 50, Z = 0.01a', fontsize=20)
plt.tight_layout()

'''
***********************************************************************
'''
# FIG 5
import matplotlib
plt.scatter(v/12700., angle, c=tair, cmap='jet', vmin=1.0, vmax=2000,
            s=4, linewidths=0, norm=matplotlib.colors.LogNorm())
clb = plt.colorbar()
clb.set_label(r'$t_{air} (s)$', fontsize=14, labelpad=-35, y=1.075, rotation=0)
plt.vlines(2380./12700., 0, 90)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'$V_{ej}/U$', fontsize=16)
plt.ylabel('$ Angle ' + ' (^{o})$', fontsize=16)
plt.xlim(1e-4, 1)  # plt.xlim(0.5,2.0)
plt.ylim(0, 90)
plt.title('CPPR = 50, Z = 0.01a', fontsize=20)
plt.tight_layout()

'''
***********************************************************************
'''

# last
plt.pcolormesh(model.xc/(10000.), model.yc /
               (10000.), stepend.mat, cmap='Pastel1')
plt.plot(stepend.xmark[ix]/(10000.), stepend.ymark[ix]/(10000.), 'ko')
plt.xlim(0, 15)
plt.ylim(-4, 4)
plt.xlabel('$x / a$', fontsize=16)
plt.ylabel('$y / a$', fontsize=16)
plt.title('CPPR = 80, Z = 0.01a', fontsize=20)
plt.tight_layout()

'''
***********************************************************************
'''
# all three of them
plt.scatter(xpos/10000., angle, c=v/12700., cmap='jet',
            vmin=0.1, vmax=0.5, s=4, linewidths=0)
plt.colorbar()
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('xpos/R', fontsize=16)
plt.ylabel('angle', fontsize=16)
plt.xlim(0.5, 1.5)
plt.ylim(0, 90)
plt.title('CPPR = 1000', fontsize=20)
plt.tight_layout()

# I need to bin the data
bins1 = np.arange(0.0, 12, 0.5)
bins1 = np.arange(0.1, 1.6, 0.1)

inds = np.digitize(xpos/10000., bins1)

# There are few erroneous data (not anymore)

data = []

for i in range(len(bins1)):
    idx = np.where(inds == i)
    data.append(np.mean(v[idx]/12700.))

data = np.array(data)
plt.plot(bins1, data, "ko")
plt.xscale('log')
plt.yscale('log')
# From Wünnemann al. 2017
plt.plot(bins1, 0.57*(bins1**-2.3), "ro")

model.closeFile()
'''
***********************************************************************
'''

tracer_idx = tracer_idx.astype('int')


plt.pcolormesh(model.xc, model.yc, step1.mat, cmap='Pastel1')

plt.plot(step1.xmark[model.tru[0].start:model.tru[0].end],
         step1.ymark[model.tru[0].start:model.tru[0].end], 'bo')
plt.plot(step1.xmark[model.tru[1].start:model.tru[1].end],
         step1.ymark[model.tru[1].start:model.tru[1].end], 'ro')
plt.plot(step1.xmark[ix], step1.ymark[ix], "ko")
plt.plot(step1.xmark[model.tru[1].end], step1.ymark[model.tru[1].end], 'go')

# find tracers fullfilling conditions in the grid (going way too slow this way, but maybe the only way?)
ikok = np.where(step1.data[0] > 2680.)
ikok2 = np.where(step1.data[0] < 2680.)

plt.plot(model.xc[ikok], model.yc[ikok], "ko")
plt.plot(model.xc[ikok2], model.yc[ikok2], "ro")

locTracers = []

for xo in range(np.shape(ikok)[1]):
    a = ikok[0][xo]
    b = ikok[1][xo]
    print xo, a, b, stepor.findTracer(a, b)

    # findTracer take x and y coordinates
    locTracers.append(step1.findTracer(a, b))

# other possibility

# get the extension of the hiresolution zone (did not finish that yet)
xhires = model.xext[1] - model.xext[0]
yhires = model.yext[1] - model.yext[0]

nx, ny = (xhires, yhires)

x = np.arange(1, nx+1)
y = np.arange(1, ny+1)
y = y[::-1]

xx, yy = np.meshgrid(x, y)


# Calculation the specific internal energy, I don't really understand this part
plt.pcolormesh(model.xc, model.yc, step1.data[2], cmap='autumn')
plt.colorbar()

ikok = np.where((step1.data[0] > 2680.) & (step1.cmc[0] > 0))
ixox = np.where((step1.data[0] < 2680.) & (stepSie.data[0] < 3.5e6) & (
    model.yc >= 1000.) & (step1.cmc[0] > 0))
ixox2 = np.where((step1.data[0] < 2680.) & (
    stepSie.data[0] < 3.5e6) & (model.yc <= 1000.) & (step1.cmc[0] > 0))
plt.plot(model.xc[ixox], model.yc[ixox], "ko")
plt.plot(model.xc[ixox2], model.yc[ixox2], "bo")
plt.plot(model.xc[ikok], model.yc[ikok], "yo")


##

plt.pcolormesh(model.xc, model.yc, stepor.mat, cmap='Pastel1')
tracer_idx = tracer_idx.astype('int')
plt.plot(stepor.xmark[tracer_idx], stepor.ymark[tracer_idx], 'ko')


# playing with VEL
stepVEL2 = model.readStep(['Den', 'TrP', 'VEL'], 29)
stepVEL3 = copy.deepcopy(stepVEL2)
Vxy = np.sqrt((stepVEL3.data[2]**2.) + (stepVEL3.data[3]**2.))
plt.pcolormesh(model.xc, model.yc, Vxy/12700., cmap='jet', vmin=0.1, vmax=2.0)
plt.colorbar()
