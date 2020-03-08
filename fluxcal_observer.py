import pandas as pd
import numpy as np

from typing import Union, Dict
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates.angles import Angle
from astroquery.jplhorizons import Horizons, Conf
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = True
iers.conf.remote_timeout = 40

ALMA = EarthLocation(
    lat=-23.0262015 * u.deg,
    lon=-67.7551257 * u.deg,
    height=5060 * u.m)

ALMA_COORD = {
    'lon': ALMA.lon.degree,
    'lat': ALMA.lat.degree,
    'elevation': ALMA.height.to_value(unit='km')}

# Add new ephemeris columns that are being returned by the JPL service.
# Workaround.
Conf.eph_columns['ObsSub-LON'] = ('a1', 'deg')
Conf.eph_columns['ObsSub-LAT'] = ('a2', 'deg')
Conf.eph_columns['SunSub-LON'] = ('a3', 'deg')
Conf.eph_columns['SunSub-LAT'] = ('a4', 'deg')

SSO_ID_DICT = {
    'Sun': 0, 'Moon': 301, 'Mars': 4,
    'Jupiter': 5, 'Uranus': 7, 'Neptune': 8,
    'Io': 501, 'Europa': 502, 'Ganymede': 503,
    'Callisto': 504
}

BAND_LIMS = {'B3': 217.45,
             'B6': 90.99,
             'B7': 61.72,
             'B9': 31.46}


def calc_ha(ra, lst):

    ha = np.degrees(
        np.math.atan2(
            np.sin(np.deg2rad(lst*15.) - np.deg2rad(ra)),
            np.cos(np.deg2rad(lst*15.) - np.deg2rad(ra))
        )
    )/15.

    return ha


def get_sso_coordinates(
        sso_name: str, epoch: Union[float, Dict[str, str]],
        raw_table: bool = False
) -> Union[SkyCoord, Table]:

    """
      Get the ICRS coordinates of a Solar System Object from JPL Horizons as a
    SkyCord object. For debuging purposes, or further information, is possible
    to get a Table object with the all the information from Horizons.

    :param epoch: jd time or epochs dict
    :param sso_name: str. Must be any of Sun, Moon, Mars, Jupter, Uranus,
        Neptune
    :param raw_table:
    :return: SkyCoord object with the SSO coordinates at the given time
    """
    if sso_name not in SSO_ID_DICT.keys():
        raise KeyError('SSO Name provided is not valid.')

    source_query = Horizons(
        id=SSO_ID_DICT[sso_name], location=ALMA_COORD, epochs=epoch,
        id_type='id'
    )
    source_table = source_query.ephemerides()
    if raw_table:
        return source_table

    source = SkyCoord(
        ra=source_table['RA'].tolist() * u.deg,
        dec=source_table['DEC'].tolist() * u.deg,
        frame='icrs', obstime=Time(source_table['datetime_jd'], format='jd'))
    return source


def get_separation_to_sso(
        coordinates: SkyCoord, sso_name: str,
        epoch: Union[float, Dict[str, str]]
) -> Angle:

    """
    Get the separation between a set of coordinates and a sso object
    :param epoch: jd time or epochs dict
    :param sso_name:
    :param coordinates: SkyCoord object
    :return: Separation to the Moon in degrees
    """

    sso_source = get_sso_coordinates(sso_name, epoch)

    return sso_source.separation(coordinates)


def get_separations_dataframe(
        main_source: SkyCoord,
        skycoord_dict: Dict[str, SkyCoord]
) -> pd.DataFrame:

    """

    :param main_source:
    :param skycoord_dict:
    :return:
    """
    try:
        cols = len(main_source)
        times = main_source.obstime.iso
    except TypeError:
        cols = len(list(skycoord_dict.values())[0])
        times = list(skycoord_dict.values())[0].obstime.iso
    tmpdata = np.zeros([cols, len(skycoord_dict)])
    columns = []

    for i, items in enumerate(skycoord_dict.items()):
        tmpdata[:, i] = main_source.separation(items[1]).arcsec
        columns.append(items[0])

    df = pd.DataFrame(
        tmpdata.copy(), columns=columns,
        index=pd.Series(times).apply(
            lambda x: pd.Timestamp(x))
    )

    if 'Jupiter' in columns:
        df['Jupiter'] -= 25.
    df['closest_object'] = df.idxmin(axis=1)
    df['closest_distance'] = df.min(axis=1)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timestamp'}, inplace=True)

    return df


def get_jovians_info(
        epoch: Union[float, Dict[str, str]], sun: SkyCoord = None,
        moon: SkyCoord = None
) -> Dict[str, pd.DataFrame]:

    """
    Returns a dictionary with information about the two Jovian's absolute flux
    calibrators, Ganymede and Callisto.

    :param epoch: jd or Dictionary as {'start': isotime, 'stop': isotime,
        'step': jpl_step}
    :param sun:
    :param moon:
    :return:
    """

    io = get_sso_coordinates('Io', epoch)
    europa = get_sso_coordinates('Europa', epoch)
    ganymede = get_sso_coordinates('Ganymede', epoch)
    callisto = get_sso_coordinates('Callisto', epoch)
    jupiter = get_sso_coordinates('Jupiter', epoch)

    if not sun:
        sun = get_sso_coordinates('Sun', epoch)
    if not moon:
        moon = get_sso_coordinates('Moon', epoch)

    ganymede_df = get_separations_dataframe(
        ganymede, {'Jupiter': jupiter, 'Io': io, 'Europa': europa,
                   'Callisto': callisto, 'Sun': sun, 'Moon': moon}
    )
    ganymede_df['altitude'] = ganymede.transform_to(
        AltAz(location=ALMA)).alt.deg
    ganymede_df['lst'] = ganymede.obstime.sidereal_time(
        'apparent', longitude=ALMA.lon).hour

    callisto_df = get_separations_dataframe(
        callisto, {'Jupiter': jupiter, 'Io': io, 'Europa': europa,
                   'Ganymede': ganymede, 'Sun': sun, 'Moon': moon}
    )
    callisto_df['altitude'] = callisto.transform_to(
        AltAz(location=ALMA)).alt.deg
    callisto_df['lst'] = callisto.obstime.sidereal_time(
        'apparent', longitude=ALMA.lon).hour

    return {'Ganymede': ganymede_df, 'Callisto': callisto_df}


def get_source_info(
        sso_name: str = None, epoch: Union[float, Dict[str, str]] = None,
        ra: Angle = None, dec: Angle = None, obstime: Time = None,
        sun: SkyCoord = None, moon: SkyCoord = None
) -> pd.DataFrame:
    """

    :param sso_name:
    :param epoch:
    :param ra:
    :param dec:
    :param sun:
    :param moon:
    :return:
    """
    if (moon and not sun) or (sun and not moon):
        raise BaseException('If a sun or moon object is provided, '
                            'both should be provided')
    if not sun:
        sun = get_sso_coordinates('Sun', epoch)
    if not moon:
        moon = get_sso_coordinates('Moon', epoch)

    if sso_name:
        coords = get_sso_coordinates(sso_name, epoch)
    else:
        coords = SkyCoord(ra, dec, frame='icrs')

    df = get_separations_dataframe(coords, {'Sun': sun, 'Moon': moon})

    if not sso_name:
        df['altitude'] = df.timestamp.apply(
            lambda x: coords.transform_to(
                AltAz(location=ALMA, obstime=Time(x.isoformat(), scale='utc'))
            ).alt.deg)
        df['lst'] = df.timestamp.apply(
            lambda x: Time(x.isoformat(), scale='utc').sidereal_time(
                'apparent', longitude=ALMA.lon).hour)
    else:
        df['altitude'] = coords.transform_to(
            AltAz(location=ALMA)).alt.deg
        df['lst'] = coords.obstime.sidereal_time(
            'apparent', longitude=ALMA.lon).hour

    return df


def df_cal_selector(x: pd.DataFrame) -> pd.Series:
    """

    :param x:
    :return:
    """
    sel_elevation = False
    sel_sun = False
    sel_moon = False
    band3 = False
    band6 = False
    band7 = False
    band9 = False

    if x['altitude'] >= 20:
        sel_elevation = True
    
    if x['Sun'] > 21 * 3600:
        sel_sun = True

    if x['Moon'] > 5 * 3600:
        sel_moon = True

    if x['closest_distance'] > BAND_LIMS['B3']:
        band3, band6, band7, band9 = True, True, True, True
    elif x['closest_distance'] > BAND_LIMS['B6']:
        band6, band7, band9 = True, True, True
    elif x['closest_distance'] > BAND_LIMS['B7']:
        band7, band9 = True, True
    elif x['closest_distance'] > BAND_LIMS['B9']:
        band9 = True

    return pd.Series(
        [sel_elevation, sel_sun, sel_moon, band3, band6, band7, band9], 
        index=['selAlt', 'selSun', 'selMoon', 'Band3', 'Band6', 'Band7',
               'Band9']
    )
