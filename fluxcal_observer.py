import numpy as np
import pandas as pd

from typing import Union, Dict
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, EarthLocation
from astroquery.jplhorizons import Horizons, Conf
from astropy.time import Time

from astropy.coordinates.angles import Angle
from astropy.coordinates import AltAz
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

JUP_RADIUS = 25.


class FluxcalObs(object):
    """ Class to handle Calibrators Monitoring.

    Parameters
    ----------
    start : string
        A start date in format 'YYYY-MM-DD'.
    stop : string
        An end date, Same format as start
    step : string
        Optional, step lengths between `start` and `stop` dates.
    """

    def __init__(self, start: str, stop: str, step: str = '15min'):

        self.epoch = {'start': start, 'stop': stop, 'step': step}
        self.sun = get_sso_coordinates('Sun', self.epoch)
        self.moon = get_sso_coordinates('Moon', self.epoch)
        self._steps = len(self.sun.obstime)

        self.main_frame = pd.concat(
            [self._create_source('Mars'), self._create_source('Uranus'),
             self._create_source('Neptune')], axis=0, sort=False)
        self.sources = ['Mars', 'Uranus', 'Neptune']

        self._io_skycoord = get_sso_coordinates('Io', self.epoch)
        self._europa_skycoord = get_sso_coordinates('Europa', self.epoch)
        self._ganymede_skycoord = get_sso_coordinates('Ganymede', self.epoch)
        self._callisto_skycoord = get_sso_coordinates('Callisto', self.epoch)
        self._jupiter_skycoord = get_sso_coordinates('Jupiter', self.epoch)

        self.add_source(
            'Ganymede', skycoord_dict={
                'Jupiter': self._jupiter_skycoord, 'Io': self._io_skycoord,
                'Europa': self._europa_skycoord,
                'Callisto': self._callisto_skycoord, 'Sun': self.sun,
                'Moon': self.moon})
        self.add_source(
            'Callisto', skycoord_dict={
                'Jupiter': self._jupiter_skycoord, 'Io': self._io_skycoord,
                'Europa': self._europa_skycoord,
                'Ganymede': self._ganymede_skycoord, 'Sun': self.sun,
                'Moon': self.moon})

    def _create_source(
            self, name: str, ra: Angle = None, dec: Angle = None,
            skycoord_dict: Dict[str, SkyCoord] = None, debug: bool = False
            ) -> pd.DataFrame:
        """Auxiliary method to create a `source` given a name and equatorial
        coordinates.

        Parameters
        ----------
        name : str
            Source's name
        ra : astropy Angle
            Source's Right Ascension as an Astropy Angle
        dec : astropy Angle
            Source's Declination as an Astropy Angle
        skycoord_dict : dict
            Optional 
        debut : boolean

        Returns
        -------
        DataFrame
        """

        if name in SSO_ID_DICT.keys():
            coords = get_sso_coordinates(
                name, self.epoch
            )
        else:
            ra = np.ones(self._steps) * ra
            dec = np.ones(self._steps) * dec
            obstime = self.sun.obstime
            coords = SkyCoord(ra, dec, frame='icrs', obstime=obstime)

        if skycoord_dict:
            df = self._build_dataframe(
                coords, skycoord_dict)
        else:
            df = self._build_dataframe(
                coords, {'Sun': self.sun, 'Moon': self.moon})

        self._prepare_coords_df(df, coords)

        df['source'] = name
        if debug:
            return df
        else:
            cols = ['timestamp', 'source', 'lst', 'ha', 'Sun', 'Moon',
                    'closest_object', 'closest_distance', 'altitude']
            return df[cols]

    @staticmethod
    def _build_dataframe(coords, skycoord_dict):
        """

        :param coords::
        :param skycoord_dict:
        :return:
        """

        rows = len(coords)
        tmpdata = np.zeros([rows, len(skycoord_dict)])
        columns = []

        for i, items in enumerate(skycoord_dict.items()):
            tmpdata[:, i] = coords.separation(items[1]).arcsec
            columns.append(items[0])

        df = pd.DataFrame(
            tmpdata.copy(), columns=columns,
            index=pd.Series(coords.obstime.iso).apply(
                lambda x: pd.Timestamp(x))
        )

        if 'Jupiter' in columns:
            df['Jupiter'] -= JUP_RADIUS
        df['closest_object'] = df.idxmin(axis=1)
        df['closest_distance'] = df.min(axis=1)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)

        return df

    @staticmethod
    def _prepare_coords_df(df: pd.DataFrame, coords: SkyCoord):
        """
           Add extra column parameters to a DataFrame produced by
        _build_dataframe. These parameters are Altitude, LST and HA, and
        they are under the columns `altitude`, `lst` and `ha`

        :param df: dataframe created by get_separations_dataframe
        :param coords: the SkyCoord object that was used to create the `df`
        """

        if df.shape[0] != len(coords):
            raise ValueError(
                'The input dataframe was not generated from the given SkyCoord '
                'object')

        df['altitude'] = coords.transform_to(
            AltAz(location=ALMA)).alt.deg
        df['lst'] = coords.obstime.sidereal_time(
            'apparent', longitude=ALMA.lon).hour
        df['ra'] = coords.ra.deg
        df['ha'] = df.apply(
            lambda x: calc_ha(x['ra'], x['lst']), axis=1)
        df.drop(['ra'], axis=1, inplace=True)

    def add_source(
            self, name: str, ra: Angle = None, dec: Angle = None,
            skycoord_dict: Dict[str, SkyCoord] = None):
        """

        :param name:
        :param ra:
        :param dec:
        :param skycoord_dict:
        """
        new_source = self._create_source(name, ra, dec, skycoord_dict)

        try:
            self.main_frame = pd.concat([self.main_frame, new_source], axis=0,
                                        sort=False)
            self.sources.append(name)
        except AssertionError:
            print('Can\'t add any more sources once the `apply_selector` '
                  'has been used')

    def apply_selector(
            self, horizon: float = 20., sun_limit: float = 21.,
            moon_limit: float = 5.):

        """

        :param horizon:
        :param sun_limit:
        :param moon_limit:
        """

        self.main_frame['selAlt'] = self.main_frame.altitude.apply(
            lambda x: x >= horizon)
        self.main_frame['selSun'] = self.main_frame.Sun.apply(
            lambda x: x >= sun_limit * 3600)
        self.main_frame['selMoon'] = self.main_frame.Moon.apply(
            lambda x: x >= moon_limit * 3600)

        band_columns = ['Band3', 'Band6', 'Band7', 'Band9']

        self.main_frame[band_columns] = self.main_frame.closest_distance.apply(
            lambda x: band_limits(x))


def calc_ha(ra: float, lst: float) -> float:
    """
      Returns the Hour Angle of an RA coordinate given an LST

    :param ra: Right ascension in degrees
    :param lst: Local sideral time in decimal hours
    :return: hour angle, in hours.
    """

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


def band_limits(x: float) -> pd.Series:

    band3, band6, band7, band9 = False, False, False, False

    if x > BAND_LIMS['B3']:
        band3, band6, band7, band9 = True, True, True, True
    elif x > BAND_LIMS['B6']:
        band6, band7, band9 = True, True, True
    elif x > BAND_LIMS['B7']:
        band7, band9 = True, True
    elif x > BAND_LIMS['B9']:
        band9 = True

    return pd.Series([band3, band6, band7, band9])
