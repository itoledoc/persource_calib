import pandas as pd, numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, solar_system_ephemeris, get_body, get_moon
from astroquery.jplhorizons import Horizons, Conf
from astropy.time import Time
from astroplan import Observer, FixedTarget
from astroplan import download_IERS_A, AltitudeConstraint
#download_IERS_A()

ALMA = EarthLocation(
    lat=-23.0262015 * u.deg, lon=-67.7551257 * u.deg, height=5060 * u.m)

ALMA_OBS = Observer(ALMA)
ALMA_COORD = {
    'lon': ALMA_OBS.location.lon.degree,
    'lat': ALMA_OBS.location.lat.degree,
    'elevation': ALMA_OBS.location.height.to_value(unit='km')}

# Add new ephemeris columns that are being returned by the JPL service
Conf.eph_columns['ObsSub-LON'] = ('a1', 'deg')
Conf.eph_columns['ObsSub-LAT'] = ('a2', 'deg')
Conf.eph_columns['SunSub-LON'] = ('a3', 'deg')
Conf.eph_columns['SunSub-LAT'] = ('a4', 'deg')

SSO_ID_DICT = {
    'Sun': 0, 'Moon': 301, 'Mars': 4, 'Jupiter': 5, 'Uranus': 7, 'Neptune': 8,
    'Io': 501, 'Europa': 502, 'Ganymede': 503, 'Callisto': 504}


def get_moon_separation(time: float, source_coordinate: SkyCoord) -> float:
    """

    :param time: jd time
    :param source_coordinate: SkyCoord object
    :return: Separation to the Moon in degrees
    """
    moon_query = Horizons(
        id=301, location=ALMA_COORD, epochs=time, id_type='id'
    )
    moon_table = moon_query.ephemerides()
    moon = SkyCoord(
        ra=moon_table['RA'][0] * u.deg, dec=moon_table['DEC'][0] * u.deg,
        frame='icrs')

    return moon.separation(source_coordinate).degree


def get_sun_separation(time: float, source_coordinate: SkyCoord) -> float:
    """

    :param time: jd time
    :param source_coordinate: SkyCoord object
    :return: Separation to the Sun in degrees
    """
    sun_query = Horizons(
        id=0, location=ALMA_COORD, epochs=time, id_type='id'
    )
    sun_table = sun_query.ephemerides()
    sun = SkyCoord(
        ra=sun_table['RA'][0] * u.deg, dec=sun_table['DEC'][0] * u.deg,
        frame='icrs')

    return sun.separation(source_coordinate).degree


def get_sso_coordinates(time: float, sso_name: str) -> SkyCoord:
    """

    :param time: jd time
    :param sso_name: str. Must be any of Sun, Moon, Mars, Jupter, Uranus,
        Neptune
    :return: SkyCoord object with the SSO coordinates at the given time
    """
    if sso_name not in SSO_ID_DICT.keys():
        raise KeyError('SSO Name provided is not valid.')

    source_query = Horizons(
        id=SSO_ID_DICT[sso_name], location=ALMA_COORD, epochs=time,
        id_type='id'
    )
    source_table = source_query.ephemerides()
    source = SkyCoord(
        ra=source_table['RA'][0] * u.deg, dec=source_table['DEC'][0] * u.deg,
        frame='icrs')
    return source


def get_jovians_windows(epoch: dict) -> object:
    """

    :param epoch: Dictionary as {'start': isotime, 'stop': isotime,
        'step': jpl_step}
    :return:
    """

    results={}

    io_query = Horizons(
        id=501, location=ALMA_COORD,
        epochs=epoch, id_type='id'
    )
    io_table = io_query.ephemerides()
    io = SkyCoord(
        ra=io_table['RA'].tolist() * u.deg,
        dec=io_table['DEC'].tolist() * u.deg, frame='icrs')

    europa_query = Horizons(
        id=502, location=ALMA_COORD,
        epochs=epoch, id_type='id'
    )
    europa_table = europa_query.ephemerides()
    europa = SkyCoord(
        ra=europa_table['RA'].tolist() * u.deg,
        dec=europa_table['DEC'].tolist() * u.deg, frame='icrs')

    ganymede_query = Horizons(
        id=503, location=ALMA_COORD,
        epochs=epoch, id_type='id'
    )
    ganymede_table = ganymede_query.ephemerides()
    ganymede = SkyCoord(
        ra=ganymede_table['RA'].tolist() * u.deg,
        dec=ganymede_table['DEC'].tolist() * u.deg, frame='icrs',
        obstime=Time(ganymede_table['datetime_jd'], format='jd'))

    callisto_query = Horizons(
        id=504, location=ALMA_COORD,
        epochs=epoch, id_type='id'
    )
    callisto_table = callisto_query.ephemerides()
    callisto = SkyCoord(
        ra=callisto_table['RA'].tolist() * u.deg,
        dec=callisto_table['DEC'].tolist() * u.deg, frame='icrs',
        obstime=Time(callisto_table['datetime_jd'], format='jd'))

    jupiter_query = Horizons(
        id=5, location=ALMA_COORD,
        epochs=epoch, id_type='id'
    )
    jupiter_table = jupiter_query.ephemerides()
    jupiter = SkyCoord(
        ra=jupiter_table['RA'].tolist() * u.deg,
        dec=jupiter_table['DEC'].tolist() * u.deg, frame='icrs')

    tmpdata = np.zeros([len(ganymede), 4])
    for i, jov in enumerate([jupiter, callisto, europa, io]):
        tmpdata[:, i] = ganymede.separation(jov).arcsec
    ganymede_df = pd.DataFrame(
        tmpdata, columns=['jupiter', 'callisto', 'europa', 'io'],
        index=pd.Series(ganymede.obstime.iso).apply(lambda x: pd.Timestamp(x)))
    ganymede_df['jupiter'] -= 25.
    ganymede_df['closest_object'] = ganymede_df.idxmin(axis=1)
    ganymede_df['closest_distance'] = ganymede_df.min(axis=1)
    ganymede_df.reset_index(inplace=True)
    ganymede_df.rename(columns={'index': 'timestamp'}, inplace=True)
    ganymede_df['limit_band'] = ganymede_df.closest_distance.apply(
        lambda x: get_available_band(x))

    tmpdata = np.zeros([len(callisto), 4])
    for i, jov in enumerate([jupiter, ganymede, europa, io]):
        tmpdata[:, i] = callisto.separation(jov).arcsec
    callisto_df = pd.DataFrame(
        tmpdata, columns=['jupiter', 'ganymede', 'europa', 'io'],
        index=pd.Series(callisto.obstime.iso).apply(lambda x: pd.Timestamp(x)))
    callisto_df['jupiter'] -= 25.
    callisto_df['closest_object'] = callisto_df.idxmin(axis=1)
    callisto_df['closest_distance'] = callisto_df.min(axis=1)
    callisto_df.reset_index(inplace=True)
    callisto_df.rename(columns={'index': 'timestamp'}, inplace=True)
    callisto_df['limit_band'] = callisto_df.closest_distance.apply(
        lambda x: get_available_band(x))

    return results


def get_available_band(sep: float) -> int:
    """

    :param sep: Separtion in arcsec
    :return: minimum band usable
    """

    band_lim_value = np.array([217.45, 90.99, 61.72, 31.46])
    band_lim_key = np.array([3, 6, 7, 9])

    bands_usable_key = band_lim_value < sep
    bands_usable = band_lim_key[bands_usable_key]
    if len(bands_usable) == 0:
        return 100
    else:
        return bands_usable.min()


def get_right_ephemeris_time(time,source):
    final_time=time.copy()
    transit_time=ALMA_OBS.target_meridian_transit_time(time, source)
    max_elev=ALMA_OBS.altaz(transit_time, source).alt.degree
    if max_elev > 42.:
        set_time=ALMA_OBS.target_set_time(time, source, horizon =40 * u.deg)
    elif max_elev > 22.:
        set_time=transit_time+pd.Timedelta("1 hour")
    else:
        set_time=ALMA_OBS.target_set_time(time, source, horizon =20 * u.deg)
    #print(set_time.iso,time.iso)
    if set_time < time - pd.Timedelta("12 hours"):
        final_time=final_time+pd.Timedelta("1 day")
    elif set_time < time - pd.Timedelta("10 min"):
        final_time=final_time+pd.Timedelta("12 hours")
    elif set_time < time + pd.Timedelta("10 min"):
        final_time=final_time+pd.Timedelta("24 hours")

    return final_time


def get_observability_range(time, source):
    transit_time = ALMA_OBS.target_meridian_transit_time(time, source)
    # print transit_time.iso
    if transit_time - time < pd.Timedelta("-12 hours"):
        transit_time = ALMA_OBS.target_meridian_transit_time(
            time + pd.Timedelta("30 min"), source)
    max_elev = ALMA_OBS.altaz(transit_time, source).alt.degree
    # print max_elev
    if max_elev > 80.:
        init_transit_time = ALMA_OBS.target_rise_time(time, source,
                                                      horizon=80 * u.deg)
        end_transit_time = ALMA_OBS.target_set_time(time, source,
                                                    horizon=80 * u.deg)
        if init_transit_time - time < pd.Timedelta("-12 hours"):
            init_transit_time = ALMA_OBS.target_rise_time(
                time + pd.Timedelta("30 min"), source, horizon=80 * u.deg)
        if end_transit_time - time < pd.Timedelta("-12 hours"):
            end_transit_time = ALMA_OBS.target_set_time(
                time + pd.Timedelta("30 min"), source, horizon=80 * u.deg)
    else:
        init_transit_time = transit_time
        end_transit_time = transit_time

    if max_elev > 42.:
        set_time = ALMA_OBS.target_set_time(time, source, horizon=40 * u.deg)
        if set_time - time < pd.Timedelta("-12 hours"):
            set_time = ALMA_OBS.target_set_time(time + pd.Timedelta("30 min"),
                                                source, horizon=40 * u.deg)
        # print set_time.is
        # if set_time < time + pd.Timedelta("10 min"):
        #    time=time+pd.Timedelta("1 day")
        #    set_time=alma.target_set_time(time, source, horizon = 40 * u.deg)

        rise_time = ALMA_OBS.target_rise_time(time, source, horizon=40 * u.deg)
        if rise_time - time < pd.Timedelta("-12 hours"):
            rise_time = ALMA_OBS.target_rise_time(time + pd.Timedelta("30 min"),
                                                  source, horizon=40 * u.deg)
        # print rise_time.iso
    elif max_elev > 22.:
        rise_time = transit_time - pd.Timedelta("1 hour")
        set_time = transit_time + pd.Timedelta("1 hour")
    else:
        set_time = ALMA_OBS.target_set_time(time, source, horizon=20 * u.deg)
        if set_time - time < pd.Timedelta("-12 hours"):
            set_time = ALMA_OBS.target_set_time(time + pd.Timedelta("30 min"),
                                                source, horizon=20 * u.deg)
        rise_time = ALMA_OBS.target_rise_time(time, source, horizon=20 * u.deg)
        if rise_time - time < pd.Timedelta("-12 hours"):
            rise_time = ALMA_OBS.target_rise_time(time + pd.Timedelta("30 min"),
                                                  source, horizon=20 * u.deg)

    return (
    [(rise_time, init_transit_time), (end_transit_time, set_time)], max_elev)


def get_sun_condition_range(time_range,source_coordinate):
    rise_time=time_range[0][0]
    final_time_range = time_range.copy()
    rise_sep=get_sun_separation(rise_time,source_coordinate)
    if rise_sep < 25 and rise_sep > 15:
        set_time=time_range[1][1]
        set_sep=get_sun_separation(set_time,source_coordinate)
        if rise_sep > 21:
            if set_sep <= 21:
                dtime=(set_time-rise_time)*(rise_sep-21.)/(rise_sep-set_sep)
                end_time=rise_time+dtime
                if end_time < time_range[0][1]:
                    #final_time_range=[(time_range[0][0],end_time)]
                    final_time_range[0]=(time_range[0][0],end_time)
                    #final_time_range[1][0]=time_range[1][0]
                    final_time_range[1]=(time_range[1][0],time_range[1][0])
                elif end_time <= time_range[1][0]:
                    #final_time_range=[(time_range[0][0],time_range[0][1])]
                    #final_time_range[1][0]=time_range[1][0]
                    final_time_range[1]=(time_range[1][0],time_range[1][0])
                else:
                    final_time_range[1]=(time_range[1][0],end_time)
        else:
            if set_sep > 21.:
                dtime=(set_time-rise_time)*(21.-rise_sep)/(set_sep-rise_sep)
                end_time=rise_time+dtime
                if end_time < time_range[0][1]:
                    final_time_range[0]=(end_time,time_range[0][1])
                elif end_time <= time_range[1][0]:
                    #final_time_range=[(time_range[1][0],time_range[1][1])]
                    final_time_range[0]=(time_range[0][1],time_range[0][1])
                    #final_time_range[0][1]=time_range[0][1]
                else:
                    #final_time_range[1][1]=[(end_time,time_range[1][1])]
                    final_time_range[0]=(time_range[0][1],time_range[0][1])
                    #final_time_range[0][1]=time_range[0][1]
                    final_time_range[1]=(end_time,time_range[1][1])
            else:
                #final_time_range=[]
                final_time_range[0]=(time_range[0][1],time_range[0][1])
                final_time_range[1]=(time_range[1][0],time_range[1][0])
    elif rise_sep <= 15:
        #final_time_range=[]
        final_time_range[0]=(time_range[0][1],time_range[0][1])
        final_time_range[1]=(time_range[1][0],time_range[1][0])
    return final_time_range


def get_moon_condition_range(time_range,source_coordinate):
    rise_time=time_range[0][0]
    final_time_range = time_range.copy()
    rise_sep=get_moon_separation(rise_time,source_coordinate)
    if rise_sep < 9:
        set_time=time_range[1][1]
        set_sep=get_moon_separation(set_time,source_coordinate)
        if rise_sep > 5.:
            if set_sep <= 5:
                dtime=(set_time-rise_time)*(rise_sep-5.)/(rise_sep-set_sep)
                end_time=rise_time+dtime
                if end_time < time_range[0][1]:
                    #final_time_range=[(time_range[0][0],end_time)]
                    final_time_range[0]=(time_range[0][0],end_time)
                    #final_time_range[1][0]=time_range[1][0]
                    final_time_range[1]=(time_range[1][0],time_range[1][0])
                elif end_time <= time_range[1][0]:
                    #final_time_range=[(time_range[0][0],time_range[0][1])]
                    #final_time_range[1][0]=time_range[1][0]
                    final_time_range[1]=(time_range[1][0],time_range[1][0])
                else:
                    final_time_range[1]=(time_range[1][0],end_time)
        else:
            if set_sep > 5.:
                dtime=(set_time-rise_time)*(5.-rise_sep)/(set_sep-rise_sep)
                end_time=rise_time+dtime
                if end_time < time_range[0][1]:
                    final_time_range[0]=(end_time,time_range[0][1])
                elif end_time <= time_range[1][0]:
                    #final_time_range=[(time_range[1][0],time_range[1][1])]
                    final_time_range[0]=(time_range[0][1],time_range[0][1])
                    #final_time_range[0][1]=time_range[0][1]
                else:
                    #final_time_range[1][1]=[(end_time,time_range[1][1])]
                    final_time_range[0]=(time_range[0][1],time_range[0][1])
                    #final_time_range[0][1]=time_range[0][1]
                    final_time_range[1]=(end_time,time_range[1][1])
            else:
                #final_time_range=[]
                final_time_range[0]=(time_range[0][1],time_range[0][1])
                final_time_range[1]=(time_range[1][0],time_range[1][0])
    return final_time_range


def get_jovians_observability(start_time,end_time):
    observability={}
    observability['Ganymede']={}
    observability['Callisto']={}
    index_ganymede=0
    index_callisto=0
    seperror=5.
    start_jovians=get_jovians_separation(start_time)
    end_jovians=get_jovians_separation(end_time)

    start_available_band_ganymede=get_available_band(start_jovians['Ganymede']['minsep']-seperror,start_jovians['Ganymede']['source'])
    end_available_band_ganymede=get_available_band(end_jovians['Ganymede']['minsep']-seperror,end_jovians['Ganymede']['source'])
    if start_available_band_ganymede == end_available_band_ganymede and start_jovians['Ganymede']['source'] == end_jovians['Ganymede']['source']:
        #if start_available_band_ganymede == 100:
        #    observability['Ganymede'][index_ganymede]=(start_time,start_time,100)
        #else:
        observability['Ganymede'][index_ganymede]=(start_time,end_time,start_available_band_ganymede)
    else:
        time = start_time
        time_list=[]
        band_list=[]
        while time < end_time:
            time_list.append(time)
            if time > start_time:
                result=get_jovians_separation(time)
                available_band=get_available_band(result['Ganymede']['minsep']-seperror,result['Ganymede']['source'])
                band_list.append(available_band)
            else:
                band_list.append(start_available_band_ganymede)
            time=time+pd.Timedelta("30 min")
        time_list.append(end_time)
        band_list.append(end_available_band_ganymede)
        band_obs = start_available_band_ganymede
        time_obs = start_time
        for i in range(len(band_list)):
            if band_obs != band_list[i]:
                if time_list[i-1] > time_obs:
                    #if band_obs == 100:
                    #    observability['Ganymede'][index_ganymede]=(time_obs,time_obs,band_obs)
                    #else:
                    observability['Ganymede'][index_ganymede]=(time_obs,time_list[i-1],band_obs)
                    index_ganymede=index_ganymede+1
                #if max(band_obs,band_list[i]) == 100:
                #    observability['Ganymede'][index_ganymede]=(time_obs,time_obs,max(band_obs,band_list[i]))
                #else:
                observability['Ganymede'][index_ganymede]=(time_obs,time_list[i],max(band_obs,band_list[i]))
                index_ganymede=index_ganymede+1
                band_obs = band_list[i]
                time_obs = time_list[i]
        if band_obs == band_list[-1] and time_list[-1] > time_obs:
            #if band_obs == 100:
            #    observability['Ganymede'][index_ganymede]=(time_obs,time_obs,band_obs)
            #else:
            observability['Ganymede'][index_ganymede]=(time_obs,time_list[-1],band_obs)
            index_ganymede=index_ganymede+1

    start_available_band_callisto=get_available_band(start_jovians['Callisto']['minsep']-seperror,start_jovians['Callisto']['source'])
    end_available_band_callisto=get_available_band(end_jovians['Callisto']['minsep']-seperror,end_jovians['Callisto']['source'])
    if start_available_band_callisto == end_available_band_callisto and start_jovians['Callisto']['source'] == end_jovians['Callisto']['source']:
        #if start_available_band_callisto == 100:
        #    observability['Callisto'][index_callisto]=(start_time,start_time,100)
        #else:
        observability['Callisto'][index_callisto]=(start_time,end_time,start_available_band_ganymede)
    else:
        time = start_time
        time_list=[]
        band_list=[]
        while time < end_time:
            time_list.append(time)
            if time > start_time:
                result=get_jovians_separation(time)
                available_band=get_available_band(result['Callisto']['minsep']-seperror,result['Callisto']['source'])
                band_list.append(available_band)
            else:
                band_list.append(start_available_band_callisto)
            time=time+pd.Timedelta("30 min")
        time_list.append(end_time)
        band_list.append(end_available_band_callisto)
        band_obs = start_available_band_callisto
        time_obs = start_time
        for i in range(len(band_list)):
            if band_obs != band_list[i]:
                if time_list[i-1] > time_obs:
                    #if band_obs == 100:
                    #    observability['Callisto'][index_ganymede]=(time_obs,time_obs,band_obs)
                    #else:
                    observability['Callisto'][index_ganymede]=(time_obs,time_list[i-1],band_obs)
                    index_ganymede=index_callisto+1
                #if max(band_obs,band_list[i]) == 100:
                #    observability['Callisto'][index_callisto]=(time_obs,time_obs,max(band_obs,band_list[i]))
                #else:
                observability['Callisto'][index_callisto]=(time_obs,time_list[i],max(band_obs,band_list[i]))
                index_callisto=index_callisto+1
                band_obs = band_list[i]
                time_obs = time_list[i]
        if band_obs == band_list[-1] and time_list[-1] > time_obs:
            #if band_obs == 100:
            #    observability['Callisto'][index_callisto]=(time_obs,time_obs,band_obs)
            #else:
            observability['Callisto'][index_callisto]=(time_obs,time_list[-1],band_obs)
            index_ganymede=index_ganymede+1
    return observability

