import pandas as pd, numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import solar_system_ephemeris, get_body, get_moon
from astropy.coordinates.angles import Angle
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


def get_sso_coordinates(
        epoch: (float, dict), sso_name: str, raw_table=False) -> SkyCoord:
    """
    Get the ICRS coordinates of a Solar System Object from JPL Horizons
    :param epoch: jd time or epochs dict
    :param sso_name: str. Must be any of Sun, Moon, Mars, Jupter, Uranus,
        Neptune
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
        epoch: (float, dict), sso_name: str,
        coordinates: SkyCoord) -> Angle:
    """
    Get the separation between a set of coordinates and a sso object
    :param epoch: jd time or epochs dict
    :param sso_name:
    :param coordinates: SkyCoord object
    :return: Separation to the Moon in degrees
    """

    sso_source = get_sso_coordinates(epoch, sso_name)

    return sso_source.separation(coordinates)


def get_separations_dataframe(
        main_source: SkyCoord, skycoord_dict: dict) -> pd.DataFrame:

    tmpdata = np.zeros([len(main_source), len(skycoord_dict)])
    columns = []

    for i, items in enumerate(skycoord_dict.items()):
        tmpdata[:, i] = main_source.separation(items[1]).arcsec
        columns.append(items[0])

    df = pd.DataFrame(
        tmpdata.copy(), columns=columns,
        index=pd.Series(main_source.obstime.iso).apply(
            lambda x: pd.Timestamp(x))
    )

    if 'Jupiter' in columns:
        df['Jupiter'] -= 25.
    df['closest_object'] = df.idxmin(axis=1)
    df['closest_distance'] = df.min(axis=1)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timestamp'}, inplace=True)

    return df


def get_jovians_info(epoch: dict, sun: SkyCoord, moon: SkyCoord) -> dict:
    """

    :param epoch: Dictionary as {'start': isotime, 'stop': isotime,
        'step': jpl_step}
    :return:
    """

    results={}

    io = get_sso_coordinates(epoch, 'Io')
    europa = get_sso_coordinates(epoch, 'Europa')
    ganymede = get_sso_coordinates(epoch, 'Ganymede')
    callisto = get_sso_coordinates(epoch, 'Callisto')
    jupiter = get_sso_coordinates(epoch, 'Jupiter')

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

    return {'Ganymede' : ganymede_df, 'Callisto': callisto_df}





