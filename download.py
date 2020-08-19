# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:04:15 2017

@author: root
"""


'''
******************************************************************************

     =========================================================================
     Subroutine to import data 


     Called from xxx

     Description                                     Programmer    Date
     ------------------------------------------------------------------
     Modified version (2.0).............................NCP  2017/12/07
    ==========================================================================
    
    The version 2.0 includes:
    - download LOLA data
    - download KAGUYA TC data
    
    ==========================================================================

******************************************************************************
'''


# Importing modules
import os
import os.path
import re
import urllib
import urllib2
import numpy as np

'''
******************************************************************************
'''


def proj(LON, LAT):
    '''
    function to define the geographic coordinate systems
    '''

    wkt = 'PROJCS["Equirectangular_MOON",GEOGCS["GCS_MOON",DATUM["D_MOON",SPHEROID["MOON_localRadius",1737400.0,0.0]],PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],'
    wkt1 = 'PARAMETER["central_meridian",' + str(LON) + '],'
    wkt2 = 'PARAMETER["standard_parallel_1",' + str(LAT) + '],'
    wkt3 = 'UNIT["Meter",1.0]]'
    wktf = wkt + wkt1 + wkt2 + wkt3
    return wktf


'''
******************************************************************************
'''


def LOLA(path_down, paths, filename):
    '''
    This scripts download all LOLA profiles contain in a zone specified by the
    maximum latitude (S&N), longitude (E&W)

    input data:
    path_down = path that contains the csv file with information on the
    location and coordinates

    paths = path to save all the profiles downloaded from the ftp

    filename = filename of the .csv file

    example:

    paths = '/var/tmp/mysshfs/bydn/projects/beyondearth/nilscp/lola/automatic/Tray2016/'
    path_down = '/work/nilscp/isaleruns/project/paper2/APR/DINNER_DOUTER/'
    filename = 'non_valid_concentric.csv'

    LOLA(path_down, paths, filename)


    '''

    data = np.genfromtxt(path_down + filename,
                         delimiter=",", skip_header=1, dtype='str')
    # get name
    name = data[:, 0]

    # delete data because we are only interesting in the first column (i.e., the names)
    del data

    # generate the name of the folders to be saved in paths
    folders = []
    for nn in name:
        nd = nn.replace(" ", "_")
        ne = nd.replace(".", "_")
        folders.append(ne)

    # create folders in paths
    for ix, f in enumerate(folders):
        path_f = paths + f

        if not os.path.exists(path_f):
            os.makedirs(path_f)

    # load one more the data
    data = np.genfromtxt(path_down + filename, delimiter=",", skip_header=1)

    # get the boundaries of the zones where LOLA data will be downloaded
    maxlat = np.around(data[:, 3], decimals=2)
    minlat = np.around(data[:, 4], decimals=2)
    westernlon = np.around(data[:, 5], decimals=2)

    easternlon = np.around(data[:, 6], decimals=2)  # -180 to 180 degrees
    easternlon2 = np.around(data[:, 2], decimals=2)  # 0 to 360 degrees

    # default url where to download the data
    default_url1 = 'http://oderest.rsl.wustl.edu/livegds?query=lolardr&results=vsi&maxlat='
    default_url2 = '&minlat='
    default_url3 = '&westernlon='
    default_url4 = '&easternlon='

    # get the lola data url for all craters
    lst_url = []

    for ix, val in np.ndenumerate(maxlat):

        tmp_url = (default_url1 + str(maxlat[ix]) + default_url2 + str(minlat[ix])
                   + default_url3 + str(westernlon[ix]) + default_url4 + str(easternlon[ix]))

        lst_url.append(tmp_url)

    # download the shape files for each crater
    for ix, t_url in enumerate(lst_url):

        patht = paths + folders[ix] + '/'
        os.chdir(patht)

        response = urllib2.urlopen(t_url)
        html = response.read()

        data = html.split('</URL>')

        lst_shp = []

        word = '_shp'
        pattern = re.compile(word)

        for line in data:

            if (pattern.search(line)):
                tmp_line = line.split('<URL>')[1]
                lst_shp.append(tmp_line)

        for shpf in lst_shp:
            fname = shpf.split('/')[-1]
            urllib.urlretrieve(shpf, patht+fname)

    # I think if I remember well that I have to reprojected the data with the help
    # of reproj_tmp.prj (to have it in local coordinates)
    filename_tmp = "reproj_tmp.prj"

    for ix, f in enumerate(folders):
        os.chdir(paths + f)

        if np.isnan(easternlon2[ix]):
            None
        else:
            wkt = proj(easternlon2[ix], maxlat[ix])
            file = open(filename_tmp, "w")
            file.write(wkt)
            file.close()


'''
***********************************************************************
'''


def kaguyalistall(longitude, flag):
    '''
    generate list of input files for all latitudes

    flag = 1: dtms
    flag = 2: orthoimages

    lbl, img = kaguyalist(255,2)

    lbl, img = kaguyalist(3,1)
    '''

    if flag == 1:

        longitude_string1 = str(longitude).zfill(3)
        longitude_string2 = str(longitude+3).zfill(3)

        default_url1 = 'http://darts.isas.jaxa.jp/pub/pds3/sln-l-tc-5-dtm-map-seamless-v2.0/' + \
            'lon' + longitude_string1 + '/data/'

        latitude = np.arange(0, 85, 3)

        lbl = []
        img = []

        for ix, lat in np.ndenumerate(latitude):
            if ix[0] == 0:
                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(3).zfill(2)
                s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                    'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')
            elif ix[0] == len(latitude) - 1:
                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(lat-3).zfill(2)
                s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                    'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')
            else:
                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(lat-3).zfill(2)
                s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                    'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')

                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(lat+3).zfill(2)
                s = 'DTM_MAPs02_S' + latitude_string1 + 'E' + longitude_string1 + \
                    'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')

    if flag == 2:

        longitude_string1 = str(longitude).zfill(3)
        longitude_string2 = str(longitude+3).zfill(3)

        default_url1 = 'http://darts.isas.jaxa.jp/pub/pds3/sln-l-tc-5-ortho-map-v2.0/' + \
            'lon' + longitude_string1 + '/data/'

        latitude = np.arange(0, 92, 3)

        lbl = []
        img = []

        for ix, lat in np.ndenumerate(latitude):
            if ix[0] == 0:
                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(3).zfill(2)
                s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                    'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')
            elif ix[0] == len(latitude) - 1:
                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(lat-3).zfill(2)
                s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                    'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')
            else:
                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(lat-3).zfill(2)
                s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                    'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')

                latitude_string1 = str(lat).zfill(2)
                latitude_string2 = str(lat+3).zfill(2)
                s = 'TCO_MAP_02_S' + latitude_string1 + 'E' + longitude_string1 + \
                    'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                lbl.append(default_url1 + s + '.lbl')
                img.append(default_url1 + s + '.img')

    return lbl, img


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


def convertWtoE(longitudeW):
    '''
    convertWtoE(14.0)
    convertWtoE(8.0)
    '''

    var = 360 - np.abs(longitudeW)

    return var


'''
***********************************************************************
'''


def kaguyalist(longitude, maxlatitude, minlatitude, flag):
    '''
    generate list of input files for all latitudes

    flag = 1: dtms
    flag = 2: orthoimages
    flag = 3: evening orthoimages (http://darts.isas.jaxa.jp/pub/pds3/sln-l-tc-5-evening-map-v4.0/)

    maxlatitude,minlatitude (positive for maxi)

    lbl, img = kaguyalist(255,2)

    lbl, img = kaguyalist(3,1)

    I think the script does not work if maxlatitude = 0

    # example for tycho crater    
    lbl, img = kaguyalist(345,-40,-46,1)
    lbl2, img2 = kaguyalist(352,-40,-46,1)

    '''

    if flag == 1:

        longitude_string1 = str(longitude).zfill(3)
        longitude_string2 = str(longitude+3).zfill(3)

        default_url1 = 'http://darts.isas.jaxa.jp/pub/pds3/sln-l-tc-5-dtm-map-seamless-v2.0/' + \
            'lon' + longitude_string1 + '/data/'

        latitude = np.arange(0, 85, 3)

        nearest_max_latitude, __ = find_nearest(latitude, abs(maxlatitude))
        nearest_min_latitude, __ = find_nearest(latitude, abs(minlatitude))

        if ((np.sign(maxlatitude) == np.sign(minlatitude)) and (maxlatitude > 0)):
            latitude2 = np.arange(nearest_min_latitude,
                                  nearest_max_latitude+1, 3)
        elif ((np.sign(maxlatitude) == np.sign(minlatitude)) and (maxlatitude < 0)):
            latitude2 = np.arange(nearest_max_latitude,
                                  nearest_min_latitude+1, 3)
        else:
            latitude2n = np.arange(3, nearest_min_latitude+1, 3)
            latitude2p = np.arange(0, nearest_max_latitude+1, 3)

        lbl = []
        img = []

        if (np.sign(maxlatitude) == np.sign(minlatitude)):

            for ix, lat in np.ndenumerate(latitude2):

                if (maxlatitude > 0):
                    if lat == 0:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(3).zfill(2)
                        s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
                    else:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(lat-3).zfill(2)
                        s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')

                elif (maxlatitude < 0):
                    if lat == 0:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(3).zfill(2)
                        s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
                    elif lat == 84:
                        None
                    else:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(lat+3).zfill(2)
                        s = 'DTM_MAPs02_S' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
        else:

            for ix, lat in np.ndenumerate(latitude2n):

                if lat == 84:
                    None
                else:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(lat+3).zfill(2)
                    s = 'DTM_MAPs02_S' + latitude_string1 + 'E' + longitude_string1 + \
                        'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')

            for ix, lat in np.ndenumerate(latitude2p):
                if lat == 0:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(3).zfill(2)
                    s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                        'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')
                else:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(lat-3).zfill(2)
                    s = 'DTM_MAPs02_N' + latitude_string1 + 'E' + longitude_string1 + \
                        'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')

    elif flag == 2:

        longitude_string1 = str(longitude).zfill(3)
        longitude_string2 = str(longitude+3).zfill(3)

        default_url1 = 'http://darts.isas.jaxa.jp/pub/pds3/sln-l-tc-5-ortho-map-v2.0/' + \
            'lon' + longitude_string1 + '/data/'

        latitude = np.arange(0, 92, 3)

        nearest_max_latitude, __ = find_nearest(latitude, abs(maxlatitude))
        nearest_min_latitude, __ = find_nearest(latitude, abs(minlatitude))

        if ((np.sign(maxlatitude) == np.sign(minlatitude)) and (maxlatitude > 0)):
            latitude2 = np.arange(nearest_min_latitude,
                                  nearest_max_latitude+1, 3)
        elif ((np.sign(maxlatitude) == np.sign(minlatitude)) and (maxlatitude < 0)):
            latitude2 = np.arange(nearest_max_latitude,
                                  nearest_min_latitude+1, 3)
        else:
            latitude2n = np.arange(3, nearest_min_latitude+1, 3)
            latitude2p = np.arange(0, nearest_max_latitude+1, 3)

        lbl = []
        img = []

        if (np.sign(maxlatitude) == np.sign(minlatitude)):

            for ix, lat in np.ndenumerate(latitude2):

                if (maxlatitude > 0):
                    if lat == 0:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(3).zfill(2)
                        s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
                    else:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(lat-3).zfill(2)
                        s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')

                elif (maxlatitude < 0):
                    if lat == 0:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(3).zfill(2)
                        s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
                    elif lat == 90:
                        None
                    else:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(lat+3).zfill(2)
                        s = 'TCO_MAP_02_S' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
        else:

            for ix, lat in np.ndenumerate(latitude2n):

                if lat == 90:
                    None
                else:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(lat+3).zfill(2)
                    s = 'TCO_MAP_02_S' + latitude_string1 + 'E' + longitude_string1 + \
                        'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')

            for ix, lat in np.ndenumerate(latitude2p):
                if lat == 0:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(3).zfill(2)
                    s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                        'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')
                else:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(lat-3).zfill(2)
                    s = 'TCO_MAP_02_N' + latitude_string1 + 'E' + longitude_string1 + \
                        'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')

    elif flag == 3:

        longitude_string1 = str(longitude).zfill(3)
        longitude_string2 = str(longitude+3).zfill(3)

        default_url1 = 'http://darts.isas.jaxa.jp/pub/pds3/sln-l-tc-5-evening-map-v4.0/' + \
            'lon' + longitude_string1 + '/data/'

        latitude = np.arange(0, 92, 3)

        nearest_max_latitude, __ = find_nearest(latitude, abs(maxlatitude))
        nearest_min_latitude, __ = find_nearest(latitude, abs(minlatitude))

        if ((np.sign(maxlatitude) == np.sign(minlatitude)) and (maxlatitude > 0)):
            latitude2 = np.arange(nearest_min_latitude,
                                  nearest_max_latitude+1, 3)
        elif ((np.sign(maxlatitude) == np.sign(minlatitude)) and (maxlatitude < 0)):
            latitude2 = np.arange(nearest_max_latitude,
                                  nearest_min_latitude+1, 3)
        else:
            latitude2n = np.arange(3, nearest_min_latitude+1, 3)
            latitude2p = np.arange(0, nearest_max_latitude+1, 3)

        lbl = []
        img = []

        if (np.sign(maxlatitude) == np.sign(minlatitude)):

            for ix, lat in np.ndenumerate(latitude2):

                if (maxlatitude > 0):
                    if lat == 0:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(3).zfill(2)
                        s = 'TCO_MAPe04_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
                    else:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(lat-3).zfill(2)
                        s = 'TCO_MAPe04_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')

                elif (maxlatitude < 0):
                    if lat == 0:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(3).zfill(2)
                        s = 'TCO_MAPe04_N' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
                    elif lat == 90:
                        None
                    else:
                        latitude_string1 = str(lat).zfill(2)
                        latitude_string2 = str(lat+3).zfill(2)
                        s = 'TCO_MAPe04_S' + latitude_string1 + 'E' + longitude_string1 + \
                            'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                        lbl.append(default_url1 + s + '.lbl')
                        img.append(default_url1 + s + '.img')
        else:

            for ix, lat in np.ndenumerate(latitude2n):

                if lat == 90:
                    None
                else:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(lat+3).zfill(2)
                    s = 'TCO_MAPe04_S' + latitude_string1 + 'E' + longitude_string1 + \
                        'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')

            for ix, lat in np.ndenumerate(latitude2p):
                if lat == 0:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(3).zfill(2)
                    s = 'TCO_MAPe04_N' + latitude_string1 + 'E' + longitude_string1 + \
                        'S' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')
                else:
                    latitude_string1 = str(lat).zfill(2)
                    latitude_string2 = str(lat-3).zfill(2)
                    s = 'TCO_MAPe04_N' + latitude_string1 + 'E' + longitude_string1 + \
                        'N' + latitude_string2 + 'E' + longitude_string2 + 'SC'
                    lbl.append(default_url1 + s + '.lbl')
                    img.append(default_url1 + s + '.img')

    return lbl, img


'''
***********************************************************************
'''


def kaguya(paths, minlon, maxlon, minlat, maxlat, flag):
    '''
    Download Kaguya data

    inputs:

    list of labels and images

    paths = '/var/tmp/mysshfs/bydn/projects/beyondearth/nilscp/Moon/kaguya/DTM/Tycho/'   
    flag = 1
    minlat = -46
    maxlat = -40
    minlon = 346
    maxlon = 352

    kaguya(paths, minlon, maxlon, minlat, maxlat, flag)

    flag = 2
    paths = '/var/tmp/mysshfs/bydn/projects/beyondearth/nilscp/Moon/kaguya/Orthoimages/Tycho/'
    kaguya(paths, minlon, maxlon, minlat, maxlat, flag)

    '''

    if not os.path.exists(paths):
        os.makedirs(paths)

    os.chdir(paths)

    longitude = np.arange(0, 358, 3)

    rminlon, __ = find_nearest(longitude, minlon)
    rmaxlon, __ = find_nearest(longitude, maxlon)

    lonarray = np.arange(rminlon, rmaxlon+1, 3)

    for ix, l in np.ndenumerate(lonarray):

        lbl, img = kaguyalist(l, maxlat, minlat, flag)

        # image
        for ix1, t_url1 in enumerate(img):
            fname1 = t_url1.split('/')[-1]
            urllib.urlretrieve(t_url1, paths+fname1)
        # labels
        for ix2, t_url2 in enumerate(lbl):
            fname2 = t_url2.split('/')[-1]
            urllib.urlretrieve(t_url2, paths+fname2)
