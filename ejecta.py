# Import basic Python modules
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

# Import local Python module
import pySALEPlot as psp

# Functions
def load_ejected_materials_properties(path_data, modelname):
    """
    Parameters
    ----------
    path_jdata : str
        path to jdata file (i.e., iSALE's output file).
    modelname : str
        name of the model.

    Returns
    -------
    tracer_idx : 1-D numpy array
        tracer id of the ejected material.
    timesteps : 1-D numpy array
        timestep at which the ejected material is first detected above the 
        threshold z.        
    v : 1-D numpy array
        Ejection velocities of all the detected ejected materials.
    angle : 1-D numpy array
        Ejection angles of all the detected ejected materials.
    xpos : 1-D numpy array
        Ejection positions of all the detected ejected materials.
    tair : 1-D numpy array
        Time in the air (in s).
    xland : 1-D numpy array
        Distance to re-impact of the surface (in m).
    n : 1-D numpy array
        number of tracers detected per timestep.
    """


    pathl = Path(path_data) / (modelname + '_ejected_material_properties.txt')
    pathl_tracers = Path(path_data) / (modelname + '_ntracers.txt')
    
    data = np.loadtxt(pathl.as_posix(), delimiter=';', comments='#')

    # extract the columns from the data
    tracer_ix = data[:, 0]
    timesteps = data[:, 1]
    v = data[:, 2]
    angle = data[:, 3]
    xpos = data[:, 4]
    tair = data[:, 5]
    xland = data[:, 6]

    n = np.loadtxt(pathl_tracers.as_posix(), delimiter=';', comments='#')

    return (tracer_ix, timesteps, v, angle, xpos, tair, xland, n)


def ejected_materials_positions(model, method, threshold):
    
    """
    Detect with the help of lagrangian tracers, the materials which leave the surface
    and are more than three saved timesteps above the height threshold specified.
    The position at wich the ejected materials are detected, tracer ids,
    timesteps, .... are returned. 
    
    Parameters
    ----------
    model : model output pySALEPlot
        model output from iSALE (after ingested in pySALEPlot).
    method : integer
        either use the 2-dot (method = 2) or 3-dot (method = 3) methods.
        Slightly different results are obtained if two or three saved timesteps 
        are used to determine the impact velocity, and ejection angle
    threshold : float
        Height above which an ejected material (described here by a tracer) is
        detected (e.g., threshold = (model.cppr[0]*model.dx) * 0.01)
    Returns
    -------
    x_tracers_pos : 2-D numpy array.
        The two or three positions of all the detected ejected materials (x-axis).
    y_tracers_pos : 2-D numpy array
        The two or three positions of all the detected ejected materials (y-axis).
    tracer_idx : 1-D numpy array
        tracer id of the ejected material.
    timestep : 1-D numpy array
        timestep at which the ejected material is first detected above the 
        threshold z.
    n : 1-D numpy array
        number of tracers detected per timestep.
    """

    # create empty arrays
    x0, x1, x2, y0, y1, y2, tracer_idx, timestep, n = (np.array([]),)*9
    
    # loop through saved timesteps
    for i in range(model.nsteps-2):

        # if equal to 0
        if i == 0:
            n = np.append(n, 0.)

        # else we read the step before i, i, i+1 and i+2 (i being a saved timestep)
        else:
            step0 = model.readStep(['Den', 'TrP'], i-1)
            step1 = model.readStep(['Den', 'TrP'], i)
            step2 = model.readStep(['Den', 'TrP'], i+1)
            step3 = model.readStep(['Den', 'TrP'], i+2)


            # select where tracer heights are higher than the diameter of the
            # projectile (needs to fill this criteria for the three saved timesteps)
            ix = np.where((step1.ymark > threshold) & (step2.ymark > threshold) & (step3.ymark > threshold)
                          & (step3.ymark > step2.ymark) & (step2.ymark > step1.ymark))[0]

            # as we do this step every saved timestep, we want to only select
            # the same tracer only one time. To do so, we compare the detected
            # tracers at a timestep with tracers that
            # previously met this this criteria. Only new tracers are appended
            # to the array tracer_idx (or calculated)
            mask = np.setdiff1d(ix, tracer_idx)
            mask = mask.astype(int)

            # For the new detected tracers, positions are calculated
            # for step0, step1, step2 and step3
            x0 = np.append(x0, step0.xmark[mask])
            x1 = np.append(x1, step1.xmark[mask])
            x2 = np.append(x2, step2.xmark[mask])
            y0 = np.append(y0, step0.ymark[mask])
            y1 = np.append(y1, step1.ymark[mask])
            y2 = np.append(y2, step2.ymark[mask])

            # append only newly detected tracers
            tracer_idx = np.append(tracer_idx, mask)

            # the timestep when they were detected is also stored
            timestep = np.append(timestep, (i,) * len(mask))

            # the numbers of tracers per timestep are also stored
            n = np.append(n, len(mask))
            
    # 2 dots method
    if method == 2:
        x = np.array((x0, x1))
        y = np.array((y0, y1))

    # 3 dots method
    elif method == 3:
        x = np.array((x0, x1, x2))
        y = np.array((y0, y1, y2))
        
    else:
        print ("the specified method does not exist. " 
               "Please select method = 2 (2-dot) or method = 3 (3-dot)")
    
    # create a numpy list containing the locations of the detected ejected
    # material for two or three saved timesteps.
    x_tracers_pos = np.swapaxes(x, 0, 1)
    y_tracers_pos = np.swapaxes(y, 0, 1)
    
    # return an array with the positions (for a 3-dot or 2-dot methods)
    # the detected tracers, when they were detected, the time between saved
    # timesteps and the number of detected tracers per saved timestep
    return (x_tracers_pos, y_tracers_pos, tracer_idx, timestep, n)

def linearfit(x, a, b):
    """
    Parameters
    ----------
    x : 1D-numpy array
        x-values.
    a : float
        Slope of the linear function.
    b : float
        constant term on the y-intercept.

    Returns
    -------
    1D-numpy array
        y-value (linear function).

    """

    return a*x + b
    
def ejected_materials_properties(x, y, tracer_idx, model, method):
    """
    calculate the ejection velocity, angle and position based on the locations
    of the ejected materials (see def ejected_materials_position)

    Parameters
    ----------
    x : 2-D numpy array
        The two or three positions of all the detected ejected materials (x-axis).
    y : 2-D numpy array
        The two or three positions of all the detected ejected materials (y-axis).
    tracer_idx : 1-D numpy array
        Corresponding tracer index.
    model : model output pySALEPlot
        model output from iSALE (after ingested in pySALEPlot).
    method : integer
        either use the 2-dot (method = 2) or 3-dot (method = 3) methods.
        Slightly different results are obtained if two or three saved timesteps 
        are used to determine the impact velocity, and ejection angle

    Returns
    -------
    v : 1-D numpy array
        Ejection velocities of all the detected ejected materials.
    angle : 1-D numpy array
        Ejection angles of all the detected ejected materials.
    xpos : 1-D numpy array
        Ejection positions of all the detected ejected materials.

    """

    # creaty empty arrays
    xpos, v, angle = (np.array([]),)*3

    # get the total number of ejected materials (tracers detected)
    n_tracers = np.shape(x)[0]

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # loop through all tracers (and their positions)
    for idx in np.arange(n_tracers):

        # for each tracer (and the two or three positions)
        # we fit a linear equation
        popt, pcov = curve_fit(linearfit, x[idx], y[idx])

        # generate a fit through points
        xs = np.linspace(np.min(x[idx]), np.max(x[idx]), 10)
        ys = linearfit(xs, *popt)

        # launch position
        pos = - popt[1]/popt[0]

        # get the dx and dy
        xmin, xmax, ymin, ymax = np.min(xs), np.max(xs), np.min(ys), np.max(ys)

        # calculate the distance with Pythagore
        L = np.sqrt(((xmax - xmin)**2.) + ((ymax - ymin)**2.))

        # calculate the velocity of the ejecta! That should depend on the method
        # if it is three dots, than it should be 3.*dt
        v = np.append(v, (L / (method*model.dtsave)))  # should be model.dt

        # calculate the angle
        teta = np.arctan((ymax - ymin) / (xmax - xmin)) * (180. / np.pi)

        angle = np.append(angle, teta)

        # calculate the original position
        xpos = np.append(xpos, pos)

    return (v, angle, xpos)
    
def wrap(model, method, thresholdf, g):
    """
    The function wrap gathers all of the functions in ejecta.py in one function
    and run all of them to infer information about the ejected materials and
    excavated properties of a crater. 

    Parameters
    ----------
    model : model output pySALEPlot
        model output from iSALE (after ingested in pySALEPlot).
    method : integer
        either use the 2-dot (method = 2) or 3-dot (method = 3) methods.
        Slightly different results are obtained if two or three saved timesteps 
        are used to determine the impact velocity, and ejection angle.
    thresholdf : float
        Relative height (normalized to radius projectile size).
    g : float
        Surface gravity (m2/s).

    Returns
    -------
    Ve : float
        volume of excavated material.
    de : float
        maximum depth of excavated material.
    De : float
        maximum distance of excavated material (equivalent to diameter).
    x_tracers : 1-D numpy array
        x-position of the tracers of all the excavated materials 
        (initial position).
    y_tracers : 1-D numpy array
        y-position of the tracers of all the excavated materials 
        (initial position).
    x_tracers_contour : 1-D numpy array
        (x) outermost contour of the tracers of all the excavated materials 
        (initial position).
    y_tracers_contour : 1-D numpy array
        (y) outermost contour of the tracers of all the excavated materials 
        (initial position).
    timesteps : 1-D numpy array
        timestep at which the ejected material is first detected above the 
        threshold z.
    tracer_idx : 1-D numpy array
        tracer id of the ejected material.
    n : 1-D numpy array
        number of tracers detected per timestep.
    v : 1-D numpy array
        Ejection velocities of all the detected ejected materials.
    angle : 1-D numpy array
        Ejection angles of all the detected ejected materials.
    xpos : 1-D numpy array
        Ejection positions of all the detected ejected materials.
    tair : 1-D numpy array
        Time in the air (in s).
    land : 1-D numpy array
        Distance to re-impact of the surface (in m).
    """

    threshold = (model.cppr[0]*model.dx) * thresholdf
    x, y, tracer_idx, timesteps, n = ejected_materials_positions(model, method, threshold)
    v, angle, xpos = ejected_materials_properties(x, y, tracer_idx, model, method)
    tracer_idx = tracer_idx.astype(int)
    initial_step = model.readStep(['Den', 'TrP'], 0)
    Ve, de, De = excavation_properties(model, initial_step, tracer_idx)
    x_tracers, y_tracers, x_tracers_contour, y_tracers_contour = excavation_profile(model, initial_step, tracer_idx)
    tair, land = blanket_thickness(v, angle, xpos, tracer_idx, g)

    return (Ve, de, De, 
            x_tracers, y_tracers, x_tracers_contour, y_tracers_contour, 
            timesteps, tracer_idx, n, v, angle, xpos,
            tair, land)


def calculate(path_jdata, path_data, method, thresholdf, g):
    """
    Save all the variables computed in the function warp in a number of text
    files (semi-color separated). 
    
    Parameters
    ----------
    path_jdata : str
        path to jdata file (i.e., iSALE's output file).
    path_data : str
        path to folder where to save .
    method : integer
        either use the 2-dot (method = 2) or 3-dot (method = 3) methods.
        Slightly different results are obtained if two or three saved timesteps 
        are used to determine the impact velocity, and ejection angle.
    thresholdf : float
        Relative height (normalized to radius projectile size).
    g : float
        Surface gravity (m2/s).

    Returns
    -------
    None.

    """
    # pathlib
    pathl = Path(path_jdata)
    modelname = pathl.parts[-2]
    print (modelname)
    paths = Path(path_data) / modelname

    model = psp.opendatfile(pathl.as_posix())
    (Ve, de, Re, x_tracers, y_tracers, x_tracers_contour, y_tracers_contour, 
     timesteps, tracer_idx, n, v, angle, xpos,
     tair, land) = wrap(model, method, thresholdf, g)

    # create plots directory if it does not exist
    if not path_data.exists():
        path_data.mkdir(parents=True)

    # save data
    header_txt = "Ve;de;De"
    
    # we need to define the name of the txt file that will be saved
    fname = modelname + '_excavation_properties.txt'
    fname_pathlib = paths / fname
    
    # depth, diameter, volume of excavated materials
    output = np.column_stack((Ve, de, Re*2.))
    np.savetxt(fname_pathlib.as_posix(), output, header=header_txt,
               delimiter=';', fmt=['%1.6e', '%1.6e', '%1.6e'])

    # x and y positions of ejected materials (Lagrangian tracers)
    fname = modelname + '_excavation_tracers_XY.txt'
    fname_pathlib = paths / fname
    header_txt = "x_tracers;y_tracers"
    output = np.column_stack((x_tracers, y_tracers))
    np.savetxt(fname_pathlib.as_posix(), output, header=header_txt,
               delimiter=';', fmt=['%1.6e', '%1.6e'])

    # x and y positions of ejected materials (Lagrangian tracers, contour)
    fname = modelname + '_excavation_tracers_XY_contour.txt'
    fname_pathlib = paths / fname
    header_txt = "x_tracers_contour;y_tracers_contour"
    output = np.column_stack((x_tracers_contour, y_tracers_contour))
    np.savetxt(fname_pathlib.as_posix(), output, header=header_txt,
               delimiter=';', fmt=['%1.6e', '%1.6e'])

    # ejected material properties
    header_txt = "tracer_idx;timesteps;v;angle;xpos;tair;xland"
    # we need to define the name of the txt file that will be saved
    fname_ej = modelname + '_ejected_material_properties.txt'
    fname_pathlib = paths / fname_ej
    output = np.column_stack((tracer_idx, timesteps, v, angle, xpos,
                              tair, land))
    np.savetxt(fname_pathlib.as_posix(), output, header=header_txt,
               delimiter=';', fmt=['%1.6e', '%1.6e', '%1.6e',
                                   '%1.6e', '%1.6e', '%1.6e', '%1.6e'])

    header_txt = "ntracers"
    # we need to define the name of the txt file that will be saved
    fname_ej = modelname + '_ntracers.txt'
    fname_pathlib = paths / fname_ej
    output = n
    np.savetxt(fname_pathlib.as_posix(), output, header=header_txt,
               delimiter=';', fmt='%1.6e')

    # close model file
    model.closeFile()

def parabolafit(x, a, b, c):
    
    """
    Parabola equation.
    
    Parameters
    ----------
    x : TYPE
        x-values.
    a : float
        parameter a in parabolic function.
    b : float
        parameter b in parabolic function.
    c : float
        parameter c in parabolic function.

    Returns
    -------
    y : 1-D numpy array
        y-value (parabolic function).

    """

    y = (a*(x**2)) + (b*x) + c

    return y


def blanket_thickness(v, angle, xpos, tracer_idx, g):
    """
    preliminary script to calculate where the material ejected will land
    (in order to calculate the thickness of the ejecta)
    
    More work is required here!!!
    
    Parameters
    ----------
    v : 1-D numpy array
        Ejection velocities (in m/s).
    angle : 1-D numpy array
        Ejection angles (in degrees).
    xpos : 1-D numpy array
        Ejection positions (m).
    tracer_idx : 1-D numpy array
        tracer index.
    g : float
        Surface gravity in m2/s.

    Returns
    -------
    tair : 1-D numpy array
        Time in the air (in s).
    land : 1-D numpy array
        Distance to re-impact of the surface (in m).

    """
    # flight time
    tair = (2. * v * np.sin(angle * (np.pi/180))) / g

    # position where it landed
    land = xpos + (v*tair*np.cos(angle * (np.pi/180.)))

    # only take ejected materials that landed outside of the crater
    #mask = np.where(land > radius)
    #idx = mask[0]
    # I dunno, this things does not work so well
    return (tair, land)
 
def excavation_properties(model, initial_step, tracer_idx):
    """
    calculate the volume, depth, and diameter of the exavated materials
    This task is done using position of tracers    

    Parameters
    ----------
    model : model output pySALEPlot
        model output from iSALE (after ingested in pySALEPlot).
    initial_step : step pySALEPlot
        Initial step (before the start of the simulation).
    tracer_idx : 1-D numpy array
        Tracer indexes (of ejected materials).

    Returns
    -------
    Ve : float
        Excavated volume.
    de : float
        Maximum excavated depth.
    De : float
        Maximum excavated distance (which translates into diameter).

    """

    # index for tracers of material below the surface (we want to avoid to
    # take tracers from the projectile)
    ix_mat1 = np.where(initial_step.ymark[tracer_idx] <= 0)

    # get the tracer id of the tracers originating below the surface at step 0
    tr_idx1 = tracer_idx[ix_mat1]

    # calculate excavated volume
    r1 = initial_step.xmark[tr_idx1]
    rii1 = initial_step.xmark[tr_idx1]+model.dx
    h = model.dy  # same values as in vimod (take that)
    R1 = r1 + ((1.) * (rii1 - r1))
    vol1 = np.pi * h * ((R1**2) - (r1**2))
    Ve = np.sum(vol1)

    # calculate maximum depth of excavation
    de = np.min(initial_step.ymark[tr_idx1])

    # calculate diameter of excavation
    De = np.max(initial_step.xmark[tr_idx1])

    # return data
    return (Ve, de, De)

def excavation_profile(model, initial_step, tracer_idx):
    """
    Extract the position of all tracers and of the outermost tracers (contour).
    Particularly interesting to plot where the ejected materials originate
    from.

    Parameters
    ----------
    model : model output pySALEPlot
        model output from iSALE (after ingested in pySALEPlot).
    initial_step : step pySALEPlot
        Initial step (before the start of the simulation).
    tracer_idx : 1-D numpy array
        Tracer indexes (of ejected materials).

    Returns
    -------
    x_tracers : 1-D numpy array
        x-position of the tracers of all the excavated materials (initial position).
    y_tracers : 1-D numpy array
        y-position of the tracers of all the excavated materials (initial position).
    x_tracers_contour : 1-D numpy array
        (x) outermost contour of the tracers of all the excavated materials (initial position).
    y_tracers_contour : 1-D numpy array
        (y) outermost contour of the tracers of all the excavated materials (initial position).
    """


    # index for tracers of material below the surface (we want to avoid to
    # take tracers from the projectile)
    ix_mat1 = np.where(initial_step.ymark[tracer_idx] <= 0)


    # tracer number for material 1 and 2
    tr_idx1 = tracer_idx[ix_mat1]
    
    x_tracers = initial_step.xmark[tr_idx1]
    y_tracers = initial_step.ymark[tr_idx1]

    # only take unique distance
    x_tracers_contour = np.unique(x_tracers)

    # get the x-index
    xidx = x_tracers_contour/(model.dx/2.)
    xidx = xidx.astype(int)

    # create empty matrix y-index
    y_tracers_contour = np.ones(len(x_tracers_contour))

    # the minimum origin emplacement of ejected materials are calculated
    # in order to get the boundary
    for i, var in np.ndenumerate(xidx):
        ii = i[0]
        idx_test = np.where((x_tracers == x_tracers_contour[ii]))
        y_tracers_contour[ii] = np.min(y_tracers[idx_test])

    return (x_tracers, y_tracers, x_tracers_contour, y_tracers_contour)
