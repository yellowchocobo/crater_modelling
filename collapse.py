# -*- coding: utf-8 -*-

'''
******************************************************************************

     =========================================================================
     Subroutine to calculate crater collapse from an analytical description
     (Grieve et al. 1984), and from the Z-model (steady state) described in 
     Croft et al. (1980) and Grieve et al. (1984).


     Called from xxx

     Description                                     Programmer    Date
     ------------------------------------------------------------------
     Original version (1.0).............................NCP  2017/07/25
     Improved version (2.0)   ..........................NCP  2017/22/11
     
     (I should add the analytical model of Melosh)
    ==========================================================================
    
    The version 2.0 includes:
    - a better description of functions
    - changed the name of some function
    - functions:
    
    
    Various parameters are calculated via an analytical description of crater
    collapse and the steady-state Z-model. Below, a list of calculated parameters
    are summarized:
    
    Final crater dimensions:
    
    Diameter rim crest: Dr
    Diameter original ground plane: Da
    Depth original ground plane to top of breccia lens: da
    Depth original ground plane to base of breccia lens: dt
    Height rim uplift above original ground plane: hr
    Volume breccia lens: Vbr
    Volume final apparent crater at original ground plane Va
    Rim crest volume apparent crater: Vapc
    Rim crest volume final true crater: Vftc
    
    
    Transient and excavated cavities:
    
    Diameter transient cavity at rim crest: Dtrc
    Diameter excavated and transient cavities at original ground plane: Dtc = De
    Depth original ground plane to floor of transient cavity: dtc
    Maximum depth excavation below original ground plane: de
    Height transient rim uplift:  htr
    rim crest volume of transient cavity: Vtrc
    Volume excavated: Ve
    Volume transient cavity at original ground plane: Vatc
    
    functions:
    
    
    ==========================================================================

******************************************************************************
'''

# load basic module of Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


'''
***********************************************************************
'''


def Dtc(dtc):
    '''
    description:

    Diameter excavated and transient cavities at original ground plane: Dtc = De
    because it is assumed to be entirely due to excavation. 

    input:
    dtc: depth from the original ground plane to the floor of the transient cavity
    '''

    var = np.sqrt(2. * (dtc**2.)) * 2.

    return var


'''
***********************************************************************
'''


def dtc(Dtc):
    '''
    description:

    depth from the original ground plane to the floor of the transient cavity: dtc
    This crater dimension is often assumed to be the depth from the original groud 
    plane to the floor of the true final crater in the autochtonuous target.

    See Grieve et al. (1984) for more information

    input:
    Dtc: The transient crater diameter (measured at the original ground plane)
    '''

    Rtc = Dtc / 2.
    var = np.sqrt((Rtc**2.)/2.)

    return var


'''
***********************************************************************
'''


def volParaboloid(r, d):
    '''
    description:

    calculate the volume of a paraboloid

    inputs:
    r as the radius of the crater (either true or apparent or transient)
    d as the depth of the crater (same as above)
    '''

    var = 0.5 * np.pi * (r**2.) * d

    return var


'''
***********************************************************************
'''


def eqParaboloid(x, a, b, c):
    '''
    description:

    equation of a paraboloid. Can be used to describe a crater profile with a
    parabola

    input:
    x: distance from the axis of symmetry

    output:
    return the height   
    '''

    return a*(x**2.) + b*(x) + c


'''
***********************************************************************
'''


def linearfit(x, a, b):
    '''
    description:
    linear fit
    '''

    return a*x + b


'''
***********************************************************************
'''

# Height transient rim uplift:  htr


def profileParaboloid(D, d):
    '''
    description:
    Derive the parameters a,b,c of a paraboloid if the diameter and depth of 
    the transient, apparent or final craters are given

    inputs:
    D: diameter    
    d: depth

    outputs:
    a, b, c, which can be used to plot the height of the parabola from known
    distances from the axis of symmetry

    examples:
    # for the transient crater
    x = np.linspace(0,Dtc/2.,1000)
    popt = transient_para(Dtc,dtc)

    y = profileParaboloid(x,*popt)


    # for the real crater
    #Dr : observed radius rim crest
    #hr : observed height of rim crest
    #dt : depth to the base of the lence of breccia

    x2 = np.linspace(0,Dr/2.,1000)    
    popt2 = transient_para(Dr,dt+hr)

    y2 = profileParaboloid(x2,*popt2)

    plt.plot(x,y,"r",linewidth=3)
    plt.plot(x2,y2,"b",linewidth=3)
    '''

    a = d / ((D/2.)**2.)
    b = 0
    c = 0  # -dtc

    return (a, b, c)


'''
***********************************************************************
'''


def deg2rad(deg):
    '''
    description:
    convert from degree to radian
    '''

    var = deg * (np.pi/180.)

    return var


'''
***********************************************************************
'''


def rad2deg(rad):
    '''
    description:
    convert from radian to degree
    '''

    var = rad * (180./np.pi)

    return var


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

    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


'''
***********************************************************************
'''


def quadratic(a, b, c):
    '''
    description:
    find the nearest value in two different arrays


    '''
    d = (b**2.) - (4.*a*c)

    if d > 0:
        r1 = (-b + np.sqrt(d)) / (2.*a)
        r2 = (-b - np.sqrt(d)) / (2.*a)
        var = (r1, r2)
    elif d == 0:
        var = -b/(2.*a)
    else:
        print 'the root is negative'
        var = np.nan
    return var


'''
***********************************************************************
'''


def htr(dtcv, slope, hr, Dr):
    '''
    description:
    reconstruct the height of the transient rim uplift (Grieve et al., 1984)
    from the observed (apparent) height of the rim and:

    different from the technique used in function analytical

    inputs:
    slope : slope outer rim
    hr: Height rim uplift above original ground plane
    Dr: Final rim-to-rim crater diameter
    dtcv = transient crater depth (can be derived from drilling)

    example:
    # Meteor crater
    dtcv = 310.
    Dr = 1156.
    hr = 67.5
    slope = 10
    htrv = htr(dtcv,slope,hr,Dr)

    plots:
    plt.plot(x,y-dtcv,"r",linewidth=3)

    y2 = linearfit(x,*popt2)
    plt.plot(x,y2,"g",linewidth=3)
    plt.hlines(0,0,Dr/2.)
    plt.hlines(htrv,0,Dr/2.)

    '''
    # calculate the transient crater diameter from the depth
    Dtcv = Dtc(dtcv)

    # Calculate the transient crater profile
    x = np.linspace(0, Dr/2., 1000)
    popt = profileParaboloid(Dtcv, dtcv)
    y = eqParaboloid(x, *popt)

    # Calculate the outer rim profile
    x2 = (Dr/2.) + (hr / (np.tan(deg2rad(slope))))
    x3 = np.linspace(0, x2*1.5, 1000.)
    y2 = x3 * np.tan(deg2rad(slope))
    x4 = x2 - x3

    # for the outer rim slope
    popt2, pcov = curve_fit(linearfit, x4, y2)

    # calculate the intersection of the transient crater profile and outer rim
    a = popt[0]
    b = popt2[0]
    c = popt2[1]

    # we get the intersection by equating the two equations
    var = quadratic(a, -b, -c-dtcv)
    varg = var[np.where(var > 0)[0][0]]

    # get the final results
    fi1, fi2 = find_nearest(x, varg)
    ht = y[fi2]-dtcv

    return ht


'''
***********************************************************************
'''


def fk(D, d):
    '''
    Equations to make easier the derivation in Grieve et al. (1984)    
    '''

    k = d / ((D/2.)**2.)

    return k


'''
***********************************************************************
'''


def fP(D, d):
    '''
    Equations to make easier the derivation in Grieve et al. (1984)    
    '''

    ktc = fk(D, d)
    p = 0.25 * (1./ktc)

    return p


'''
***********************************************************************
'''


def analytical(Dr, hr, dt, slope):
    '''
    description:

    input:
    Diameter rim crest: Dr
    Depth original ground plane to base of breccia lens: dt
    Height rim uplift above original ground plane: hr
    an outer rim slope assumed to be about 10 degrees (from terrestrial and
    lunar observations, Pike 1977 and more, see Grieve and Garvin 1984): slope


    example (Meteor Crater) from Grieve et al. 1984:

    Dr = 1156.
    hr = 67.5
    dt = 310.
    slope = 10.

    Dtcv, htrv, Vtrc, Vfrc, Vbr = analytical(Dr,hr,dt,slope)

    '''

    # parabola of the actual crater
    #x = np.linspace(0,Dr/2.,1000)
    #popt = profileParaboloid(Dr,dt+hr)
    #y = eqParaboloid(x,*popt)

    # parabola of the transient crater
    Dtcv = Dtc(dt)
    #popt2 = profileParaboloid(Dtcv,dt)
    #y2 = eqParaboloid(x,*popt2)

    # calculating ktc and kfc (transient crater and final crater)
    ktc = fk(Dtcv, dt)
    #kfc = fk(Dr,dt+hr)

    ptc = fP(Dtcv, dt)
    pfc = fP(Dr, dt+hr)

    # calculating the slope
    m = np.tan(deg2rad(slope))
    b = (dt + hr) + (Dr/2.) * m

    # calculating the transient rim to rim crater radius
    #Rtrc_p = (m + (((m*m)+ 4.*b*ktc)**0.5)) / (2.*ktc)
    Rtrc_m = (m - (((m*m) + 4.*b*ktc)**0.5)) / (2.*ktc)
    Rtrc = np.abs(Rtrc_m)

    # calculating the transient rim height
    htrv = ktc * (Rtrc**2.) - dt

    # calculating the volume of the transient crater (rim-rim)
    Vtrc = 2. * np.pi * ptc * ((dt + htrv)**2.)

    # calculating the volume of the final crater (rim-rim)
    Vfrc = 2. * np.pi * pfc * ((dt + hr)**2.)

    # volume of breccia why is it not Vfrc - vtrc?
    Vbr = Vfrc - Vtrc

    # plt.plot(x,y-dt,"r",linewidth=3)
    # plt.hlines(0,0,1.1*(Dr/2.),'k',linewidth=3)
    # plt.hlines(htrv,0,1.1*(Dr/2.),'k',linewidth=3)
    # plt.plot(x,y2-dt,"b",linewidth=3)

    return Dtcv, htrv, Vtrc, Vfrc, Vbr


'''
***********************************************************************
'''


def profileZmodel(Re, EDOZ, Z, teta_rad):
    '''
    EDOZ HAS TO BE POSITIVE!!!

    # Meteor Crater
    Re = 438.0
    EDOZ = 60.
    dv = delta2(Re,EDOZ)
    teta= np.linspace(0,90. + dv,2000.)
    teta_rad = deg2rad(teta)
    De = Re*2.
    Z = 2.8
    X, Y = profileZmodel(Re,EDOZ,Z,teta_rad)

    teta= np.linspace(0,160.,1000.)
    teta_rad = deg2rad(teta)

    R0, __, __ = Zmodel(Re,EDOZ,Z,deg2rad(50.))
    R = (R0) * ((1. - np.cos(deg2rad(50.)))**ff)
    X, Y = hingeZmodel(R,teta_rad)
    plt.plot(X,-Y,"k",linewidth=3)

    '''
    # calculation of angle delta
    R0, __, __ = Zmodel(Re, EDOZ, Z, teta_rad)

    ff = 1. / (Z-2.)

    # Calculate R
    # should I use R or R0!!? R makes more sense but I don't get the right Re! we get
    R = R0 * ((1. - np.cos(teta_rad))**ff)
    # a difference of EDOZ/2 (if we do expected from O = zmodel - np.max(Y)), O is half of the EDOZ.

    # Calculate X and Y
    X = R * np.sin(teta_rad)

    c = R * np.cos(teta_rad)
    d = (1. - np.cos(teta_rad))
    e = (c) * (d ** ff)
    Y = e + EDOZ

    return X, Y


'''
***********************************************************************
'''


def hingeZmodel(R, teta_rad, EDOZ, Z):
    '''
    I don't think it is used :)
    '''
    X = R * np.sin(teta_rad)
    ff = 1. / (Z-2.)
    c = R * np.cos(teta_rad)
    d = (1. - np.cos(teta_rad))
    e = (c) * (d ** ff)
    Y = e + EDOZ

    return X, Y


'''
***********************************************************************
'''


def delta2(Re, EDOZ):
    '''
    Equations to make easier the derivation in Grieve et al. (1984)    
    '''

    var = rad2deg(np.arctan(EDOZ/Re))

    return var


'''
***********************************************************************
'''


def Zmodel(Re, EDOZ, Z, teta_rad):
    '''
    EDOZ HAS TO BE POSITIVE!!!

    teta= np.linspace(0,160.,1000.)
    teta_rad = deg2rad(teta)


    examples: 
    # Meteor Crater
    Re = 438.0
    EDOZ = 60.
    dv = delta2(Re,EDOZ)
    teta= np.linspace(0,90. + dv,2000.)
    teta_rad = deg2rad(teta)
    De = Re*2.
    Z = 2.8
    Zmodel(Re,EDOZ,Z,teta_rad)


    # Brent Crater
    De = 3112
    Re = De/2.
    EDOZ = 220.
    dv = delta2(Re,EDOZ)
    teta= np.linspace(0,90. + dv,2000.)
    teta_rad = deg2rad(teta)
    Z = 2.7
    Zmodel(Re,EDOZ,Z,teta_rad)

    '''
    # calculation of angle delta
    delta = np.arctan(EDOZ/Re)  # 1

    a = np.cos(delta)
    b = 1. + np.sin(delta)

    # calculation of an exponent
    ff = 1. / (Z-2.)

    # Calculate R0
    R0 = Re / (a * (b**ff))

    # Calculate de
    c = Re * (Z-2.)
    d_exp = (1-Z) / (Z-2.)
    d = (Z-1.)**d_exp
    e = np.cos(delta)
    f = (1. + np.sin(delta))
    g = f ** ff

    de = EDOZ + ((c * d) / (e * g))
    # can also be calculated as:
    # Ym = R0 * (Z-2.) * ((Z-1.)**d_exp) + EDOZ

    # Ve

    #h = R0 **3.
    i = (Z-2.) / (Z+1.)
    #j = b**((Z+1.)/(Z-2.))
    #k = (np.pi/3.) * EDOZ * (Re**2.)
    #l = (h * i * j) + k
    #Ve = (2/3.) * np.pi * l

    # Ve2

    m = EDOZ / Re
    n = 2. * i
    o = a**3.
    p = (b / o)
    q = m + (n*p)
    Ve2 = (np.pi/3.) * (Re**3.) * q

    return R0, de, Ve2


'''
***********************************************************************
'''


def pol2cart(R, teta_rad):
    x = R * np.cos(teta_rad)
    y = R * np.sin(teta_rad)
    return(x, y)

#xtest, ytest = pol2cart(R0, teta_rad)


'''
***********************************************************************

# input Meteor crater
Dr = 1156.
hr = 67.5
dt = 310.
De = 876.
Re = De /2.
EDOZ = 60.
dv = delta2(Re,EDOZ)
teta= np.linspace(0,90. + dv,2000.)
teta_rad = deg2rad(teta)
Z = 2.8

X, Y = profileZmodel(Re,EDOZ,Z,teta_rad)
plt.plot(X,-Y,'b',linewidth=3)

plt.xlim(0,Re)
plt.ylim(-dt,0)



# with dt ~ dtc
dtcv = dtc(De)
x2 = np.linspace(0,De/2.,1000)
popt2 = profileParaboloid(De,dtcv)
y2 = eqParaboloid(x2,*popt2)
plt.plot(x2,y2-dt,"r",linewidth=3)


***********************************************************************


# reproducing figure 5 in Grieve and Garvin (1984)

# Meteor Crater
Re = 438.0
EDOZ = 60.
dv = delta2(Re,EDOZ)
teta= np.linspace(0,90. + dv,2000.)
teta_rad = deg2rad(teta)
De = Re*2.
Z = np.linspace(2.5,3.0,200)
Ve = np.ones(len(Z))
EDOZ = np.arange(30.,100.,10.)

for iy, yy in np.ndenumerate(EDOZ):
    for ix, zz in np.ndenumerate(Z):
        
        __, __, Vetmp = Zmodel_func(Re,yy,zz,teta_rad)
        Ve[ix] = Vetmp
    plt.plot(Z,Ve/1e9,"o")
    

plt.xlim(2.5,3.0)
plt.ylim(0.03,0.08) 
plt.hlines(0.05,2.5,3.0,linewidth=1)  
plt.hlines(0.06,2.5,3.0,linewidth=1)
plt.vlines(2.7,0.03,0.08,linewidth=1)  
plt.vlines(2.9,0.03,0.08,linewidth=1)





***********************************************************************
'''
