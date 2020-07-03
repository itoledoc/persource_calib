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

JUP_RADIUS = 25. #I think SSR use 20.


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
             self._create_source('Neptune')], axis=0, sort=False,
            ignore_index=True)
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
            kindofsource: list = [], skycoord_dict: Dict[str, SkyCoord] = None,debug: bool = False
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
        kindofsource: list
            List of integer for B3,B6,B7, and B9, giving the kind of source.
            If the source not need to be observed in a band the integer should be 0.
            If it is a secondary amplitude calibrator in a band the integer should be 2.
            If it is a source already observed but with pending ampcal in a band the integer should be 3.
            If it is source with pending observation in a band the integer should be 4.
            For primary amp cal the integer is 1. By default if the list is empty the source
            is assumed need to be observed in all bands so it is [4,4,4,4].
        skycoord_dict : dict
            Optional 
        debut : boolean

        Returns
        -------
        DataFrame
        """
        #Definition for the kind of source could be
        # Primary amplitude calibrator kind = 1
        # Secondary amplitude calibrator kind = 2
        # QSO need a secondary amplitud calibrator kind = 3
        # QSO need a observation kind = 4

        if name in SSO_ID_DICT.keys():
            coords = get_sso_coordinates(
                name, self.epoch
            )
            # Primary amplitude calibrator kind = 1
            list_kindofsource=[1,1,1,1]
        else:
            ra = np.ones(self._steps) * ra
            dec = np.ones(self._steps) * dec
            obstime = self.sun.obstime
            coords = SkyCoord(ra, dec, frame='icrs', obstime=obstime)
            if len(kindofsource) == 0:
                # QSO need observation kind = 4
                list_kindofsource = [4, 4, 4, 4]
            else:
                # Secondary amplitude calibrator kind = 2
                list_kindofsource=[]
                for k in kindofsource:
                    if k < 5:
                        list_kindofsource.append(k)
                    else:
                        list_kindofsource.append(4)
                for i in range(len(list_kindofsource),4):
                    list_kindofsource.append(4)

        if skycoord_dict:
            df = self._build_dataframe(
                coords, skycoord_dict)
        else:
            df = self._build_dataframe(
                coords, {'Sun': self.sun, 'Moon': self.moon})

        self._prepare_coords_df(df, coords)
        df[['kind_b3', 'kind_b6', 'kind_b7', 'kind_b9']] = df.apply(lambda x: pd.Series(list_kindofsource[:4]), axis=1)
        df['source'] = name
        if debug:
            return df
        else:
            cols = ['timestamp', 'source', 'lst', 'ha', 'dec', 'Sun', 'Moon',
                    'closest_object', 'closest_distance', 'altitude','kind_b3','kind_b6','kind_b7','kind_b9']
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
        df['dec'] = coords.dec.deg
        df['ha'] = df.apply(
            lambda x: calc_ha(x['ra'], x['lst']), axis=1)
        df.drop(['ra'], axis=1, inplace=True)

    def add_source(
            self, name: str, ra: Angle = None, dec: Angle = None,
            kindofsource: list = [], skycoord_dict: Dict[str, SkyCoord] = None):
        """

        :param name:
        :param ra:
        :param dec:
        :param kindofsource:
            List of integer for B3,B6,B7, and B9, giving the kind of source.
            If the source not need to be observed in a band the integer should be 0.
            If it is a secondary amplitude calibrator in a band the integer should be 2.
            If it is a already source observed but with pending ampcal in a band the integer should be 3.
            If it is source with pending observation in a band the integer should be 4.
            For primary amp cal the integer is 1. By default if the list is empty the source
            is assumed need to be observed in all bands so it is [4,4,4,4].
        :param skycoord_dict:
        """
        new_source = self._create_source(name, ra, dec, kindofsource, skycoord_dict)

        try:
            self.main_frame = pd.concat([self.main_frame, new_source], axis=0,
                                        sort=False, ignore_index=True)
            self.sources.append(name)
        except AssertionError:
            print('Can\'t add any more sources once the `apply_selector` '
                  'has been used')

    def apply_selector(
            self, horizon: float = 25.,transit_limit: float =86., sun_limit: float = 21.,
            moon_limit: float = 5.):

        """

        :param horizon:
        :param transit_limit:
        :param sun_limit:
        :param moon_limit:
        """
        #From SSR the source selection to observe consider source over 24 deg in elevation (horizon)
        # and below 87. deg in elevation (transit_limit)
        horizon_sso=31. #From SSR is 30. deg in elevation
        transit_limit_sso=79. #From SSR is 80. deg in elevation
#        self.main_frame['selAlt'] = self.main_frame.altitude.apply(
#            lambda x: x >= horizon)
        self.main_frame['selAlt'] = self.main_frame.apply(
            lambda x: x['altitude'] >= horizon if x['kind_b3'] !=1 else
            x['altitude'] >= horizon_sso,axis=1
            )
#        self.main_frame['selTran'] = self.main_frame.altitude.apply(
#            lambda x: x <= transit_limit)
        self.main_frame['selTran'] = self.main_frame.apply(
            lambda x: x['altitude'] <= transit_limit if x['kind_b3'] != 1 else
            x['altitude'] <= transit_limit_sso,axis=1
            )
        self.main_frame['selSun'] = self.main_frame.Sun.apply(
            lambda x: x >= sun_limit * 3600)
        self.main_frame['selMoon'] = self.main_frame.Moon.apply(
            lambda x: x >= moon_limit * 3600)

        band_columns = ['Band3', 'Band6', 'Band7', 'Band9']

        self.main_frame[band_columns] = self.main_frame.closest_distance.apply(
            lambda x: band_limits(x))

    def apply_soft_selector(
            self, horizon: float = 40.,transit_limit: float =80.):

        """

        :param horizon:
        :param transit_limit:
        """
        #We can include soft constraints selection to consider source over 40 deg in elevation (horizon)
        # and below 80. deg in elevation (transit_limit). Specially for secondary amp. cal.
        #Aditionally for those source at high declination, abs(DEC-ALMA_LATITUDE) >= 49
        # or max elevation of 41 deg we consider range of time of +/- 1 hour from the
        # transit as the best observation windows.
        horizon_sso=40. #From SSR is 30. deg in elevation
        transit_limit_sso=79. #From SSR is 80. deg in elevation
        #horizon=40. # or at least 30. the same as SSOs
        #transit_limit=79. # The same as SSOs
        self.main_frame['selSoftAlt'] = self.main_frame.apply(
            lambda x: abs(x['ha']) <= 1.0  if abs(x['dec']-ALMA_COORD['lat']) >= 49. and x['kind_b3'] !=1 else
            x['altitude'] >= horizon if x['kind_b3'] != 1 else
            x['altitude'] >= horizon_sso,axis=1
            )
        self.main_frame['selSoftTran'] = self.main_frame.apply(
            lambda x: x['altitude'] <= transit_limit  if x['kind_b3'] !=1 else
            x['altitude'] <= transit_limit_sso,axis=1
            )

    def add_ampcal_condition(self):
        #To run this procedure we should first run apply_selector
        # and apply_soft_selector. We need to include the case when
        # one or both procedures are not applied to the data.
        caldata_hardconst = self.main_frame.query(
            'selAlt == True and selTran == True and selSun == True and selMoon == True'
        ).copy()
        caldata_hardconst[['B3_second_ampcal', 'B6_second_ampcal', 'B7_second_ampcal']] = caldata_hardconst[
            ['kind_b3', 'kind_b6', 'kind_b7']].applymap(lambda x: 1 if x == 2 else 0)
        caldata_softconst = caldata_hardconst.query(
            'selSoftAlt == True and selSoftTran == True'
        ).copy()
        #Primary amplitud calibrator
        primary_ampcal_availability_hardconst=caldata_hardconst.query('kind_b3 == 1').groupby(
            ['timestamp']
        ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}
        ).reset_index().rename(columns={'Band3':'B3_prim_ampcal','Band6':'B6_prim_ampcal','Band7':'B7_prim_ampcal'})

        primary_ampcal_availability_softconst=caldata_softconst.query('kind_b3 == 1').groupby(
            ['timestamp']
        ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}
        ).reset_index().rename(columns={'Band3':'B3_soft_prim_ampcal','Band6':'B6_soft_prim_ampcal','Band7':'B7_soft_prim_ampcal'})

        #Secondary amplitud calibrator
        secondary_ampcal_availability_hardconst=caldata_hardconst.query('kind_b3 != 1').groupby(
            ['timestamp']
        ).aggregate({'B3_second_ampcal': sum, 'B6_second_ampcal': sum, 'B7_second_ampcal': sum}
        ).reset_index()
        secondary_ampcal_availability_softconst=caldata_softconst.query('kind_b3 != 1').groupby(
            ['timestamp']
        ).aggregate({'B3_second_ampcal': sum, 'B6_second_ampcal': sum, 'B7_second_ampcal': sum}
        ).reset_index().rename(columns={'B3_second_ampcal':'B3_soft_second_ampcal','B6_second_ampcal':'B6_soft_second_ampcal','B7_second_ampcal':'B7_soft_second_ampcal'})

        self.main_frame['selAll']=self.main_frame.apply(lambda x: 1 if x['selAlt'] == True and x['selTran'] == True and x['selSun'] == True and x['selMoon'] == True else 0,axis=1)
        self.main_frame['selSoftAll']=self.main_frame.apply(lambda x: 1 if x['selSoftAlt'] == True and x['selSoftTran'] == True and x['selSun'] == True and x['selMoon'] == True else 0,axis=1)
        self.main_frame=self.main_frame.merge(primary_ampcal_availability_hardconst,on='timestamp',how='left').fillna(0.0)
        self.main_frame=self.main_frame.merge(primary_ampcal_availability_softconst,on='timestamp',how='left').fillna(0.0)
        self.main_frame=self.main_frame.merge(secondary_ampcal_availability_hardconst,on='timestamp',how='left').fillna(0.0)
        self.main_frame=self.main_frame.merge(secondary_ampcal_availability_softconst,on='timestamp',how='left').fillna(0.0)


    def get_observation_windows(self, soft_const: bool = False):

        """
        :param soft_const:
        """
        if soft_const:
            find_windows = self.main_frame.query(
                'selSoftAlt == True and selSoftTran == True and selSun == True and selMoon == True'
            ).groupby(
                ['source', 'timestamp']
            ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}).sort_values(by=['source', 'timestamp'])
        else:
            find_windows = self.main_frame.query(
                'selAlt == True and selTran == True and selSun == True and selMoon == True'
            ).groupby(
                ['source','timestamp']
            ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}).sort_values(by=['source', 'timestamp'])

        a = find_windows.reset_index().sort_values(by=['source', 'timestamp'])
        a = a.merge((find_windows - find_windows.shift(1)).reset_index(), on=['source', 'timestamp'], suffixes=('', '_diff'))
        a = a.merge((find_windows - find_windows.shift(-1)).reset_index(), on=['source', 'timestamp'],suffixes=('', '_diff_prev'))
        a['timediff'] = a.timestamp.diff().apply(lambda x: x.total_seconds() / 3600.0).fillna(-24.)
        a['timediff_prev'] = a.timestamp.diff(periods=-1).apply(lambda x: x.total_seconds() / 3600.0).fillna(24.)
        kind_source = self.main_frame[['source', 'kind_b3', 'kind_b6', 'kind_b7', 'kind_b9']].drop_duplicates()

        startwindow = a.query('Band3 == True and (Band3_diff == 1 or abs(timediff) > 0.25)')[
            ['source', 'timestamp']].reset_index(drop=True).copy()
        endwindow = a.query('Band3 == True and (Band3_diff_prev == 1 or abs(timediff_prev) > 0.25)')[
            ['source', 'timestamp']].reset_index(drop=True).copy()
        windows = startwindow.merge(endwindow[['timestamp']], left_index=True, right_index=True).rename(
            columns={'timestamp_x': 'start', 'timestamp_y': 'end'})
        windows['band'] = 3
        windows = windows.merge(kind_source[['source', 'kind_b3']].rename(columns={'kind_b3': 'kind'}), on='source')

        startwindow_b6 = a.query('Band6 == True and (Band6_diff == 1 or abs(timediff) > 0.25)')[
            ['source', 'timestamp']].reset_index(drop=True).copy()
        endwindow_b6 = a.query('Band6 == True and (Band6_diff_prev == 1 or abs(timediff_prev) > 0.25)')[
            ['source', 'timestamp']].reset_index(drop=True).copy()
        windows_b6 = startwindow_b6.merge(endwindow_b6[['timestamp']], left_index=True, right_index=True).rename(
            columns={'timestamp_x': 'start', 'timestamp_y': 'end'})
        windows_b6['band'] = 6
        windows_b6 = windows_b6.merge(kind_source[['source', 'kind_b6']].rename(columns={'kind_b6': 'kind'}),
                                      on='source')
        windows = windows.append(windows_b6)
        startwindow_b7 = a.query('Band7 == True and (Band7_diff == 1 or abs(timediff) > 0.25)')[
            ['source', 'timestamp']].reset_index(drop=True).copy()
        endwindow_b7 = a.query('Band7 == True and (Band7_diff_prev == 1 or abs(timediff_prev) > 0.25)')[
            ['source', 'timestamp']].reset_index(drop=True).copy()
        windows_b7 = startwindow_b6.merge(endwindow_b6[['timestamp']], left_index=True, right_index=True).rename(
            columns={'timestamp_x': 'start', 'timestamp_y': 'end'})
        windows_b7['band'] = 7
        windows_b7 = windows_b7.merge(kind_source[['source', 'kind_b7']].rename(columns={'kind_b7': 'kind'}),
                                      on='source')
        return windows.append(windows_b7)

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
