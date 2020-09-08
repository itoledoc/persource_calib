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

JUP_RADIUS = 25.  # I think SSR use 20.


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
        self.simulation_frame = None
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
            kindofsource: list = None,
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
        kindofsource: list
            List of integer for B3,B6,B7, and B9, giving the kind of source.
            If the source not need to be observed in a band the integer should
            be 0.
            If it is a secondary amplitude calibrator in a band the integer
            should be 2.
            If it is a source already observed but with pending ampcal in a band
            the integer should be 3.
            If it is source with pending observation in a band the integer
            should be 4.
            For primary amp cal the integer is 1. By default if the list is
            empty the source is assumed need to be observed in all bands so it
            is [4,4,4,4].
        skycoord_dict : dict
            Optional
        debug : boolean

        Returns
        -------
        DataFrame
        """
        # Definition for the kind of source could be
        # No need observation kind = 0
        # Primary amplitude calibrator kind = 1
        # Secondary amplitude calibrator kind = 2
        # QSO need a secondary amplitud calibrator kind = 3
        # QSO need a observation kind = 4.

        if name in SSO_ID_DICT.keys():
            coords = get_sso_coordinates(
                name, self.epoch
            )
            # Primary amplitude calibrator kind = 1
            kindofsource = [1, 1, 1, 1]

        else:
            ra = np.ones(self._steps) * ra
            dec = np.ones(self._steps) * dec
            obstime = self.sun.obstime
            coords = SkyCoord(ra, dec, frame='icrs', obstime=obstime)
            if kindofsource is None:
                kindofsource = [4, 4, 4, 4]

        if skycoord_dict:
            df = self._build_dataframe(
                coords, skycoord_dict)
        else:
            df = self._build_dataframe(
                coords, {'Sun': self.sun, 'Moon': self.moon})
        self._prepare_coords_df(df, coords)

        # check that kindofsource
        df[['kind_b3', 'kind_b6', 'kind_b7', 'kind_b9']] = df.apply(
            lambda x: pd.Series(kindofsource), axis=1)
        df['source'] = name

        if debug:
            return df

        else:
            cols = ['timestamp', 'source', 'lst', 'ha', 'dec', 'Sun', 'Moon',
                    'closest_object', 'closest_distance', 'altitude', 'kind_b3',
                    'kind_b6', 'kind_b7', 'kind_b9']
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
            kindofsource: list = None,
            skycoord_dict: Dict[str, SkyCoord] = None):
        """

        :param name:
        :param ra:
        :param dec:
        :param kindofsource:
            List of integer for B3,B6,B7, and B9, giving the kind of source.
            If the source not need to be observed in a band the integer should
            be 0.
            If it is a secondary amplitude calibrator in a band the integer
            should be 2.
            If it is a already source observed but with pending ampcal in a
            band the integer should be 3.
            If it is source with pending observation in a band the integer
            should be 4.
            For primary amp cal the integer is 1. By default if the list is
            empty the source is assumed need to be observed in all bands so it
            is [4,4,4,4].
        :param skycoord_dict:
        """
        new_source = self._create_source(
            name, ra, dec, kindofsource, skycoord_dict)

        try:
            self.main_frame = pd.concat(
                [self.main_frame, new_source], axis=0, sort=False,
                ignore_index=True)
            self.sources.append(name)
        except AssertionError:
            print('Can\'t add any more sources once the `apply_selector` '
                  'has been used')

    def apply_selector(
            self, horizon: float = 25., transit_limit: float = 86.,
            sun_limit: float = 21., moon_limit: float = 5.):

        """

        :param horizon:
        :param transit_limit:
        :param sun_limit:
        :param moon_limit:
        """
        # From SSR the source selection to observe consider source over 24 deg
        # in elevation (horizon) and below 87. deg in elevation (transit_limit)
        horizon_sso = 31.  # From SSR is 30. deg in elevation
        transit_limit_sso = 79.  # From SSR is 80. deg in elevation

        self.main_frame['selAlt'] = self.main_frame.apply(
            lambda x: x['altitude'] >= horizon if x['kind_b3'] != 1 else
            x['altitude'] >= horizon_sso, axis=1
            )

        self.main_frame['selTran'] = self.main_frame.apply(
            lambda x: x['altitude'] <= transit_limit if x['kind_b3'] != 1 else
            x['altitude'] <= transit_limit_sso, axis=1
            )
        self.main_frame['selSun'] = self.main_frame.Sun.apply(
            lambda x: x >= sun_limit * 3600)
        self.main_frame['selMoon'] = self.main_frame.Moon.apply(
            lambda x: x >= moon_limit * 3600)
        self.main_frame['selAll'] = self.main_frame.apply(
            lambda x: True if (x['selAlt'] and x['selTran'] and x['selSun'] and
                               x['selMoon']) else False,
            axis=1)

        band_columns = ['Band3', 'Band6', 'Band7', 'Band9']

        self.main_frame[band_columns] = self.main_frame.closest_distance.apply(
            lambda x: band_limits(x))

    def apply_soft_selector(
            self, horizon: float = 40., transit_limit: float = 80.):

        """
        We can include soft constraints selection to consider source over 40 deg
        in elevation (horizon) and below 80. deg in elevation (transit_limit).
        Specially for secondary amp. cal. Aditionally for those source at high
        declination, abs(DEC-ALMA_LATITUDE) >= 49 or max elevation of 41 deg we
        consider range of time of +/- 1 hour from the transit as the best
        observation windows.

        :param horizon:
        :param transit_limit:
        """

        horizon_sso = 40.  # From SSR is 30. deg in elevation
        transit_limit_sso = 79.  # From SSR is 80. deg in elevation

        self.main_frame['selSoftAlt'] = self.main_frame.apply(
            lambda x: abs(x['ha']) <= 1.0 if
            abs(x['dec']-ALMA_COORD['lat']) >= 49. and x['kind_b3'] != 1 else
            x['altitude'] >= horizon if x['kind_b3'] != 1 else
            x['altitude'] >= horizon_sso, axis=1
            )
        self.main_frame['selSoftTran'] = self.main_frame.apply(
            lambda x: x['altitude'] <= transit_limit if x['kind_b3'] != 1 else
            x['altitude'] <= transit_limit_sso, axis=1
            )
        self.main_frame['selSoftAll'] = self.main_frame.apply(
            lambda x: True if x['selSoftAlt'] and x['selSoftTran'] and
            x['selAll'] else False, axis=1)

    def ampcal_condition(self, simulate: bool = False):
        # To run this procedure we should first run apply_selector
        # and apply_soft_selector. We need to include the case when
        # one or both procedures are not applied to the data.
        if simulate:
            caldata_hardconst = self.simulation_frame.query(
                'selAll == True'
            ).copy()
        else:
            caldata_hardconst = self.main_frame.query(
                'selAll == True'
            ).copy()
        caldata_hardconst[['B3_second_ampcal', 'B6_second_ampcal', 'B7_second_ampcal']] = caldata_hardconst[
            ['kind_b3', 'kind_b6', 'kind_b7']].applymap(lambda x: 1 if x == 2 else 0)
        caldata_softconst = caldata_hardconst.query(
            'selSoftAll == True'
        ).copy()
        # Primary amplitud calibrator
        primary_ampcal_availability_hardconst = caldata_hardconst.query('kind_b3 == 1').groupby(
            ['timestamp']
        ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}
                    ).reset_index().rename(
            columns={'Band3': 'B3_prim_ampcal', 'Band6': 'B6_prim_ampcal', 'Band7': 'B7_prim_ampcal'})

        primary_ampcal_availability_softconst = caldata_softconst.query('kind_b3 == 1').groupby(
            ['timestamp']
        ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}
                    ).reset_index().rename(
            columns={'Band3': 'B3_soft_prim_ampcal', 'Band6': 'B6_soft_prim_ampcal', 'Band7': 'B7_soft_prim_ampcal'})

        # Secondary amplitud calibrator
        secondary_ampcal_availability_hardconst = caldata_hardconst.query('kind_b3 != 1').groupby(
            ['timestamp']
        ).aggregate({'B3_second_ampcal': sum, 'B6_second_ampcal': sum, 'B7_second_ampcal': sum}
                    ).reset_index()
        secondary_ampcal_availability_softconst = caldata_softconst.query('kind_b3 != 1').groupby(
            ['timestamp']
        ).aggregate({'B3_second_ampcal': sum, 'B6_second_ampcal': sum, 'B7_second_ampcal': sum}
                    ).reset_index().rename(
            columns={'B3_second_ampcal': 'B3_soft_second_ampcal', 'B6_second_ampcal': 'B6_soft_second_ampcal',
                     'B7_second_ampcal': 'B7_soft_second_ampcal'})
        ampcal_availability = primary_ampcal_availability_hardconst.copy()
        ampcal_availability = ampcal_availability.merge(primary_ampcal_availability_softconst, on='timestamp',
                                                        how='outer').fillna(0.0)
        ampcal_availability = ampcal_availability.merge(secondary_ampcal_availability_hardconst, on='timestamp',
                                                        how='outer').fillna(0.0)
        ampcal_availability = ampcal_availability.merge(secondary_ampcal_availability_softconst, on='timestamp',
                                                        how='outer').fillna(0.0)
        return ampcal_availability

    def add_ampcal_condition(self, simulate: bool = False):
        # To run this procedure we should first run apply_selector
        # and apply_soft_selector. We need to include the case when
        # one or both procedures are not applied to the data.
        if simulate:
            caldata_hardconst = self.simulation_frame.query(
                'selAll == True'
            ).copy()
        else:
            caldata_hardconst = self.main_frame.query(
                'selAll == True'
            ).copy()
        caldata_hardconst[['B3_second_ampcal', 'B6_second_ampcal', 'B7_second_ampcal']] = caldata_hardconst[
            ['kind_b3', 'kind_b6', 'kind_b7']].applymap(lambda x: 1 if x == 2 else 0)
        caldata_softconst = caldata_hardconst.query(
            'selSoftAll == True'
        ).copy()
        # Primary amplitud calibrator
        primary_ampcal_availability_hardconst = caldata_hardconst.query('kind_b3 == 1').groupby(
            ['timestamp']
        ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}
                    ).reset_index().rename(
            columns={'Band3': 'B3_prim_ampcal', 'Band6': 'B6_prim_ampcal', 'Band7': 'B7_prim_ampcal'})

        primary_ampcal_availability_softconst = caldata_softconst.query('kind_b3 == 1').groupby(
            ['timestamp']
        ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}
                    ).reset_index().rename(
            columns={'Band3': 'B3_soft_prim_ampcal', 'Band6': 'B6_soft_prim_ampcal', 'Band7': 'B7_soft_prim_ampcal'})

        # Secondary amplitud calibrator
        secondary_ampcal_availability_hardconst = caldata_hardconst.query('kind_b3 != 1').groupby(
            ['timestamp']
        ).aggregate({'B3_second_ampcal': sum, 'B6_second_ampcal': sum, 'B7_second_ampcal': sum}
                    ).reset_index()
        secondary_ampcal_availability_softconst = caldata_softconst.query('kind_b3 != 1').groupby(
            ['timestamp']
        ).aggregate({'B3_second_ampcal': sum, 'B6_second_ampcal': sum, 'B7_second_ampcal': sum}
                    ).reset_index().rename(
            columns={'B3_second_ampcal': 'B3_soft_second_ampcal', 'B6_second_ampcal': 'B6_soft_second_ampcal',
                     'B7_second_ampcal': 'B7_soft_second_ampcal'})

        if simulate:
            self.simulation_frame = self.simulation_frame.merge(primary_ampcal_availability_hardconst, on='timestamp',
                                                                how='left').fillna(0.0)
            self.simulation_frame = self.simulation_frame.merge(primary_ampcal_availability_softconst, on='timestamp',
                                                                how='left').fillna(0.0)
            self.simulation_frame = self.simulation_frame.merge(secondary_ampcal_availability_hardconst, on='timestamp',
                                                                how='left').fillna(0.0)
            self.simulation_frame = self.simulation_frame.merge(secondary_ampcal_availability_softconst, on='timestamp',
                                                                how='left').fillna(0.0)
        else:
            self.main_frame = self.main_frame.merge(primary_ampcal_availability_hardconst, on='timestamp',
                                                    how='left').fillna(0.0)
            self.main_frame = self.main_frame.merge(primary_ampcal_availability_softconst, on='timestamp',
                                                    how='left').fillna(0.0)
            self.main_frame = self.main_frame.merge(secondary_ampcal_availability_hardconst, on='timestamp',
                                                    how='left').fillna(0.0)
            self.main_frame = self.main_frame.merge(secondary_ampcal_availability_softconst, on='timestamp',
                                                    how='left').fillna(0.0)

    def get_source_with_ampcal(self, ampcal_softconst: bool = True, source_softconst: bool = True,
                               min_num_with_ampcal_sample: int = 4, simulate: bool = False):
        if simulate:
            a = self.simulation_frame.copy()
        else:
            a = self.main_frame.copy()

        if ampcal_softconst:
            a[['B3_with_prim_ampcal', 'B6_with_prim_ampcal', 'B7_with_prim_ampcal']] = a[
                ['B3_soft_prim_ampcal', 'B6_soft_prim_ampcal', 'B7_soft_prim_ampcal']].applymap(
                lambda x: 1 if x > 0 else 0)
            a[['B3_with_second_ampcal', 'B6_with_second_ampcal', 'B7_with_second_ampcal']] = a[
                ['B3_soft_second_ampcal', 'B6_soft_second_ampcal', 'B7_soft_second_ampcal']].applymap(
                lambda x: 1 if x > 0 else 0)
        else:
            a[['B3_with_prim_ampcal', 'B6_with_prim_ampcal', 'B7_with_prim_ampcal']] = a[
                ['B3_prim_ampcal', 'B6_prim_ampcal', 'B7_prim_ampcal']].applymap(lambda x: 1 if x > 0 else 0)
            a[['B3_with_second_ampcal', 'B6_with_second_ampcal', 'B7_with_second_ampcal']] = a[
                ['B3_second_ampcal', 'B6_second_ampcal', 'B7_second_ampcal']].applymap(lambda x: 1 if x > 0 else 0)
        a['B3_with_ampcal'] = a.B3_with_prim_ampcal + a.B3_with_second_ampcal
        a['B6_with_ampcal'] = a.B6_with_prim_ampcal + a.B6_with_second_ampcal
        a['B7_with_ampcal'] = a.B7_with_prim_ampcal + a.B7_with_second_ampcal
        a[['B3_with_ampcal', 'B6_with_ampcal', 'B7_with_ampcal']] = a[
            ['B3_with_ampcal', 'B6_with_ampcal', 'B7_with_ampcal']].applymap(lambda x: 1 if x > 0 else 0)
        if source_softconst:
            source_with_ampcal = a.query('kind_b3 !=1 and selAll == 1 and selSoftAll == 1').groupby(
                ['source']
            ).aggregate(
                {'kind_b3': min, 'kind_b6': min, 'kind_b7': min, 'B3_with_prim_ampcal': sum, 'B6_with_prim_ampcal': sum,
                 'B7_with_prim_ampcal': sum, 'B3_with_ampcal': sum, 'B6_with_ampcal': sum,
                 'B7_with_ampcal': sum}).reset_index()
        else:
            source_with_ampcal = a.query('kind_b3 !=1 and selAll == 1').groupby(
                ['source']
            ).aggregate(
                {'kind_b3': min, 'kind_b6': min, 'kind_b7': min, 'B3_with_prim_ampcal': sum, 'B6_with_prim_ampcal': sum,
                 'B7_with_prim_ampcal': sum, 'B3_with_ampcal': sum, 'B6_with_ampcal': sum,
                 'B7_with_ampcal': sum}).reset_index()
        source_with_ampcal[
            ['B3_with_prim_ampcal', 'B6_with_prim_ampcal', 'B7_with_prim_ampcal', 'B3_with_ampcal', 'B6_with_ampcal',
             'B7_with_ampcal']] = source_with_ampcal[
            ['B3_with_prim_ampcal', 'B6_with_prim_ampcal', 'B7_with_prim_ampcal', 'B3_with_ampcal', 'B6_with_ampcal',
             'B7_with_ampcal']].applymap(lambda x: True if x > min_num_with_ampcal_sample else False)
        return source_with_ampcal

    def source_to_observe(self, simulate: bool = False, list_source_to_observe: Dict[str, list] = {}):
        # Example of list_source_to_observe is {'b3':["s1","s2","s3"]}
        if simulate:
            a = self.simulation_frame.copy()
        else:
            a = self.main_frame.copy()
        # source_with_ampcal = self.get_source_with_ampcal(ampcal_softconst=True, source_softconst=False,simulate = simulate)
        if 'b3' in list_source_to_observe:
            list_source_to_observe_b3 = list_source_to_observe['b3']
        else:
            list_source_to_observe_b3 = a.query(
                'kind_b3 == 3 or kind_b3 == 4').source.tolist()
        if 'b6' in list_source_to_observe:
            list_source_to_observe_b6 = list_source_to_observe['b6']
        else:
            list_source_to_observe_b6 = a.query(
                'kind_b6 == 3 or kind_b6 == 4').source.tolist()
        if 'b7' in list_source_to_observe:
            list_source_to_observe_b7 = list_source_to_observe['b7']
        else:
            list_source_to_observe_b7 = a.query(
                'kind_b7 == 3 or kind_b7 == 4').source.tolist()
        # list_source_to_observe_b9=source_with_ampcal.query('(kind_b9 == 3 and B9_with_ampcal) or kind_b9 == 4').source.tolist()

        a['B3_source_to_observe'] = a.apply(
            lambda x: 1 if x['source'] in list_source_to_observe_b3 and x['selAll'] else 0, axis=1)
        a['B6_source_to_observe'] = a.apply(
            lambda x: 1 if x['source'] in list_source_to_observe_b6 and x['selAll'] else 0, axis=1)
        a['B7_source_to_observe'] = a.apply(
            lambda x: 1 if x['source'] in list_source_to_observe_b7 and x['selAll'] else 0, axis=1)
        a['B3_soft_source_to_observe'] = a.apply(
            lambda x: 1 if x['source'] in list_source_to_observe_b3 and x['selSoftAll'] else 0, axis=1)
        a['B6_soft_source_to_observe'] = a.apply(
            lambda x: 1 if x['source'] in list_source_to_observe_b6 and x['selSoftAll'] else 0, axis=1)
        a['B7_soft_source_to_observe'] = a.apply(
            lambda x: 1 if x['source'] in list_source_to_observe_b7 and x['selSoftAll'] else 0, axis=1)
        available_source_to_observe = a.groupby(
            ['timestamp']
        ).aggregate({'B3_source_to_observe': sum, 'B6_source_to_observe': sum, 'B7_source_to_observe': sum,
                     'B3_soft_source_to_observe': sum, 'B6_soft_source_to_observe': sum,
                     'B7_soft_source_to_observe': sum}).reset_index()
        return available_source_to_observe

    def add_source_to_observe(self, simulate: bool = False):
        available_source_to_observe = self.source_to_observe(simulate=simulate)
        if simulate:
            self.simulation_frame = self.simulation_frame.merge(available_source_to_observe, on='timestamp',
                                                                how='left').fillna(0.0)
        else:
            self.main_frame = self.main_frame.merge(available_source_to_observe, on='timestamp', how='left').fillna(0.0)

    def create_simulation(self):
        self.simulation_frame = self.main_frame.copy()

    def change_kind_in_simulation(self, source_with_changes_dict: Dict[int, Dict] = {}):
        # Example for source_with_changes_dict { 2: {'kind_b3': ["s1","s2","s3"]}}
        for k in source_with_changes_dict:
            for b in source_with_changes_dict[k]:
                list_of_source = source_with_changes_dict[k][b]
                self.simulation_frame[[b]] = self.simulation_frame.apply(
                    lambda x: k if x['source'] in list_of_source else x[b], axis=1)

    def observing_plan_max_peak(self, simulate: bool = False, peak_uncertainty: int = 0, min_timestamp=None,
                                delta_timestamp=pd.Timedelta('1day')):
        # makes observing plan grouping sources. Higher priority '0' is the group
        # with more sources that can be calibrated (with either primary or secondary)
        observe_timestamp = {'timestamp': [], 'peak_obs_value': [], 'final_peak_obs': [], 'num_obs': [],
                             'conditions': [], 'source_list': [], 'prim_ampcal_list': [], 'second_ampcal_list': []}
        if simulate:
            a = self.simulation_frame.copy()
        else:
            a = self.main_frame.copy()
        if min_timestamp == None:
            init_timestamp = a.timestamp.min() - pd.Timedelta('1s')
        else:
            init_timestamp = min_timestamp
        last_timestamp = init_timestamp + delta_timestamp
        list_source_to_observe = a.query(
            'kind_b3 == 3  or kind_b3 == 4').source.unique().tolist()
        list_source_all = list_source_to_observe
        ampcal_cond = self.ampcal_condition(simulate=simulate).copy()
        ampcal_list_timestamp = ampcal_cond.query(
            '(B3_soft_prim_ampcal > 0 or B3_soft_second_ampcal > 0) and timestamp > @init_timestamp and timestamp <= @last_timestamp').timestamp.tolist()
        prim_ampcal_list_timestamp = ampcal_cond.query('B3_soft_prim_ampcal > 0').timestamp.tolist()
        source_to_observe = self.source_to_observe(simulate=simulate)
        max_value = 1
        # data=source_to_observe.query('timestamp in @ampcal_list_timestamp')
        # max_value=data[['B3_soft_source_to_observe']].max().values[0]
        # timestamplist=data.query('B3_soft_source_to_observe == @max_value').timestamp.tolist()[:1]
        # observed_list=a.query('timestamp in @timestamplist and source in @list_source_to_observe and selAll == True').source.unique().tolist()
        # listaux = [s for s in list_source_to_observe if s not in observed_list]
        # list_source_to_observe=listaux
        # print(timestamplist,max_value)
        # observe_timestamp['timestamp'].append(timestamplist[0])
        # observe_timestamp['max_obs_value'].append(max_value)
        # observe_timestamp['num_obs'].append(len(observed_list))
        # observe_timestamp['conditions'].append("optimal")
        # observe_timestamp['source_list'].append(observed_list)

        while (max_value > 0):
            # source_to_observe=self.source_to_observe(simulate = simulate,list_source_to_observe={'b3':list_source_to_observe})
            data = source_to_observe.query('timestamp in @ampcal_list_timestamp')
            max_value = data[['B3_soft_source_to_observe']].max().values[0]
            if max_value > 0:
                value = max_value - peak_uncertainty
                timestamplist = data.query('B3_soft_source_to_observe >= @value').timestamp.tolist()[:1]
                if timestamplist[0] in prim_ampcal_list_timestamp:
                    observe_timestamp['conditions'].append("optimal prim_ampcal")
                else:
                    observe_timestamp['conditions'].append("optimal second_ampcal")
            else:
                max_value = data[['B3_source_to_observe']].max().values[0]
                if max_value > 0:
                    if max_value > peak_uncertainty:
                        value = max_value - peak_uncertainty
                    else:
                        value = max_value
                    timestamplist = data.query('B3_source_to_observe >= @value').timestamp.tolist()[:1]
                    if timestamplist[0] in prim_ampcal_list_timestamp:
                        observe_timestamp['conditions'].append("non optimal prim_ampcal")
                    else:
                        observe_timestamp['conditions'].append("non optimal second_ampcal")
                else:
                    data = source_to_observe.query('timestamp > @init_timestamp and timestamp <= @last_timestamp')
                    max_value = data[['B3_source_to_observe']].max().values[0]
                    if max_value > peak_uncertainty:
                        value = max_value - peak_uncertainty
                    else:
                        value = max_value
                    timestamplist = data.query('B3_source_to_observe >= @value').timestamp.tolist()[:1]
                    if max_value > 0:
                        observe_timestamp['conditions'].append("non ampcal")
                    else:
                        observe_timestamp['conditions'].append("non observable")

            observed_list = a.query(
                'timestamp in @timestamplist and source in @list_source_to_observe and selAll == True').source.unique().tolist()
            listaux = [s for s in list_source_to_observe if s not in observed_list]
            list_source_to_observe = listaux
            all_observed_list = a.query(
                'timestamp in @timestamplist and source in @list_source_all and selSoftAll == True').source.unique().tolist()
            for s in observed_list:
                if s not in all_observed_list:
                    all_observed_list.append(s)
            observe_timestamp['timestamp'].append(timestamplist[0])
            observe_timestamp['peak_obs_value'].append(max_value)
            observe_timestamp['final_peak_obs'].append(len(observed_list))
            observe_timestamp['num_obs'].append(len(all_observed_list))
            prim_ampcal_list = a.query(
                'kind_b3 == 1 and timestamp == @timestamplist[0] and selSoftAll and Band3').source.unique().tolist()
            second_ampcal_list = a.query(
                'kind_b3 == 2 and timestamp == @timestamplist[0] and selSoftAll').source.unique().tolist()
            observe_timestamp['prim_ampcal_list'].append(prim_ampcal_list)
            observe_timestamp['second_ampcal_list'].append(second_ampcal_list)
            print("Peak value: %d" % (max_value))
            print("Observed list: %s" % (all_observed_list))
            print("Source to observe: %s" % (list_source_to_observe))
            print("Timestamp: %s" % (timestamplist[0]))

            if max_value > 0:
                if observe_timestamp['conditions'][-1] == "non ampcal":
                    observe_timestamp['source_list'].append(observed_list)
                else:
                    observe_timestamp['source_list'].append(all_observed_list)
                source_to_observe = self.source_to_observe(simulate=simulate,
                                                           list_source_to_observe={'b3': list_source_to_observe})
            else:
                observe_timestamp['source_list'].append(list_source_to_observe)
        observing_plan = pd.DataFrame(observe_timestamp,
                                      columns=['timestamp', 'peak_obs_value', 'final_peak_obs', 'num_obs', 'conditions',
                                               'source_list', 'prim_ampcal_list', 'second_ampcal_list']).sort_values(
            by='timestamp')
        return observing_plan

    def observing_plan_by_source_peak(self, simulate: bool = False, peak_uncertainty: int = 0, min_timestamp=None,
                                      delta_timestamp=pd.Timedelta('1day')):
        #Makes observing plan  grouping sources for the mayor peaks within the observing windows
        #of each source. So group the sources in a different timestamps that can be calibrated
        #with either primary or secondary amplitude calibrators.
        #The output table have the following information
        #peak_count, num_orig_source, source_list, prim_ampcal_list, second_ampcal_list, num_obs, conditions
        if simulate:
            a = self.simulation_frame.copy()
        else:
            a = self.main_frame.copy()
        if min_timestamp == None:
            init_timestamp = a.timestamp.min() - pd.Timedelta('1s')
        else:
            init_timestamp = min_timestamp
        last_timestamp = init_timestamp + delta_timestamp

        # ampcal_cond: Numbers of ampcal available by timestamp
        ampcal_cond = self.ampcal_condition(simulate=simulate).copy()
        # Add ampcal_cond to the main frame
        a = a.merge(ampcal_cond, on='timestamp', how='left').fillna(0.0)

        # ampcal_list_timestamp=ampcal_cond.query('(B3_soft_prim_ampcal > 0 or
        #                                           B3_soft_second_ampcal > 0) and
        #                                           timestamp > @init_timestamp and
        #                                           timestamp <= @last_timestamp').timestamp.tolist()
        # Timestamps with available primary ampcal
        prim_ampcal_list_timestamp = ampcal_cond.query('B3_soft_prim_ampcal > 0').timestamp.tolist()

        # source_to_observe: Numbers of source needing observations available by timestamp
        source_to_observe = self.source_to_observe(simulate=simulate)
        # Add source_to_observe to the main frame
        a = a.merge(source_to_observe, on='timestamp', how='left').fillna(0.0)

        # Select from the main frame source nedding observations in B3 and available from SSR conditions
        a_source_to_observe = a.query(
            '(kind_b3 == 3  or kind_b3 == 4) and selAll == True'
        )
        # Filter those timestamp with an available ampcal in optimal conditions and within the time windows defined
        a_ampcal_cond = a_source_to_observe.query(
            '(B3_soft_prim_ampcal > 0 or B3_soft_second_ampcal > 0) and '
            'timestamp > @init_timestamp and timestamp <= @last_timestamp')

        # Select from the filtered main frame those timestamp with the source observable in optimal conditions
        a_optimal_cond = a_ampcal_cond.query('selSoftAll == True')

        # Got the peak of sources available to be observed with calibrations for each source observility time
        # window with optimal conditions
        a_max_counts = a_optimal_cond.loc[a_optimal_cond.groupby(['source'])['B3_soft_source_to_observe'].idxmax()]
        # Filter the first timestamp by source in time
        a_timestamp = a_max_counts.loc[a_max_counts.groupby(['source'])['timestamp'].idxmin()]

        # Define the final dataframe results with the peak of source needing observation in optimal
        # conditions, grouping by timestamp. Including the info of the number of sources where the
        # peak is found, for the observavility time window with optimal condition for those sources
        a_final_optimal_cond = a_timestamp.query('B3_soft_source_to_observe > 0').groupby(
            ['timestamp', 'B3_soft_source_to_observe']
        )[['source']].aggregate(lambda x: len(x.unique().tolist())).reset_index().rename(
            columns={'B3_soft_source_to_observe': 'peak_count', 'source': 'num_orig_source'})

        # Adding to the final dataframe results the sources available to be observed with SSR conditions
        # including an amp cal
        a_final_optimal_cond = a_final_optimal_cond.merge(a_ampcal_cond.groupby(
            ['timestamp']
        )[['source']].aggregate(lambda x: x.unique().tolist()).rename(
            columns={'source': 'source_list'}).reset_index().drop_duplicates(subset=['timestamp'], keep='last')
                                                          , on='timestamp', how='left')
        prim_ampcal = a.query('kind_b3 == 1 and timestamp in @a_final_optimal_cond.timestamp.tolist()'
                              ' and selSoftAll and Band3'
                              ).groupby(
            ['timestamp']
        )[['source']].aggregate(lambda x: x.unique().tolist()).reset_index().rename(
            columns={'source': 'prim_ampcal_list'}).drop_duplicates(subset=['timestamp'], keep='last')
        a_final_optimal_cond = a_final_optimal_cond.merge(prim_ampcal, on='timestamp', how='left')

        second_ampcal = a.query('kind_b3 == 2 and timestamp in @a_final_optimal_cond.timestamp.tolist()'
                                ' and selSoftAll'
                                ).groupby(
            ['timestamp']
        )[['source']].aggregate(lambda x: x.unique().tolist()).reset_index().rename(
            columns={'source': 'second_ampcal_list'}).drop_duplicates(subset=['timestamp'])
        a_final_optimal_cond = a_final_optimal_cond.merge(second_ampcal, on='timestamp', how='left')

        a_final_optimal_cond['num_obs'] = a_final_optimal_cond.source_list.apply(lambda x: len(x))
        a_final_optimal_cond['conditions'] = a_final_optimal_cond.timestamp.apply(
            lambda x: "optimal prim_ampcal" if x in prim_ampcal_list_timestamp else "optimal second_ampcal")
        # List of source included in  the observing plan
        list_to_observe = a_ampcal_cond.query(
            'timestamp in @a_final_optimal_cond.timestamp.tolist()').source.unique().tolist()
        print("Sources to observe: ", list_to_observe)
        # List of source not included in  the observing plan but with ampcal
        list_pending_to_observe = a_ampcal_cond.query('source not in @list_to_observe').source.unique().tolist()
        print("Sources pending to observe: ", list_pending_to_observe)
        # List of source without ampcal or out of the observavility time window defined
        list_non_ampcal = a_source_to_observe.query(
            'source not in @list_to_observe and source not in @list_pending_to_observe').source.unique().tolist()
        print("Source with non ampcal: ", list_non_ampcal)
        # List of source non observable
        list_non_observable = a.query(
            '(kind_b3 == 3  or kind_b3 == 4)'
            'and source not in @list_to_observe '
            'and source not in @list_pending_to_observe '
            'and source not in @list_non_ampcal').source.unique().tolist()
        print("Source non observable: ", list_non_observable)
        return a_final_optimal_cond

        # def run_simulation(self):

    def observing_plan_by_source_peak_prim_calib(self, simulate: bool = False, peak_uncertainty: int = 0,
                                                 min_timestamp=None, delta_timestamp=pd.Timedelta('1day')):
        # Makes observing plan  grouping sources for the mayor peaks within the observing windows
        # of each source. So group the sources in a different timestamps that can be calibrated
        # with first a primary and then if needed a secondary amplitude calibrator.
        # The output table have the following information
        # peak_count, num_orig_source, source_list, prim_ampcal_list, second_ampcal_list, num_obs, conditions

        if simulate:
            a = self.simulation_frame.copy()
        else:
            a = self.main_frame.copy()
        if min_timestamp == None:
            init_timestamp = a.timestamp.min() - pd.Timedelta('1s')
        else:
            init_timestamp = min_timestamp
        last_timestamp = init_timestamp + delta_timestamp

        # ampcal_cond: Numbers of ampcal available by timestamp
        ampcal_cond = self.ampcal_condition(simulate=simulate).copy()
        # Add ampcal_cond to the main frame
        a = a.merge(ampcal_cond, on='timestamp', how='left').fillna(0.0)

        # ampcal_list_timestamp=ampcal_cond.query('(B3_soft_prim_ampcal > 0 or
        #                                           B3_soft_second_ampcal > 0) and
        #                                           timestamp > @init_timestamp and
        #                                           timestamp <= @last_timestamp').timestamp.tolist()
        # Timestamps with available primary ampcal
        prim_ampcal_list_timestamp = ampcal_cond.query('B3_soft_prim_ampcal > 0').timestamp.tolist()

        # source_to_observe: Numbers of source needing observations available by timestamp
        source_to_observe = self.source_to_observe(simulate=simulate)
        # Add source_to_observe to the main frame
        a = a.merge(source_to_observe, on='timestamp', how='left').fillna(0.0)

        # Select from the main frame source nedding observations in B3 and available from SSR conditions
        a_source_to_observe = a.query(
            '(kind_b3 == 3  or kind_b3 == 4) and selAll == True'
        )
        # Filter those timestamp with an available ampcal in optimal conditions and within the time windows defined
        a_ampcal_cond = a_source_to_observe.query(
            '(B3_soft_prim_ampcal > 0) and timestamp > @init_timestamp and timestamp <= @last_timestamp')

        # Select from the filtered main frame those timestamp with the source observable in optimal conditions
        a_optimal_cond = a_ampcal_cond.query('selSoftAll == True')

        prim_ampcal_list_source = a_optimal_cond.source.unique().tolist()

        a_ampcal_cond = a_source_to_observe.query(
            '(B3_soft_prim_ampcal > 0 or (B3_soft_second_ampcal > 0 and source not in @prim_ampcal_list_source)) and timestamp > @init_timestamp and timestamp <= @last_timestamp')
        a_optimal_cond = a_ampcal_cond.query('selSoftAll == True')

        # Got the peak of sources available to be observed with calibrations for each source observility time
        # window with optimal conditions
        a_max_counts = a_optimal_cond.loc[a_optimal_cond.groupby(['source'])['B3_soft_source_to_observe'].idxmax()]
        # Filter the first timestamp by source in time
        a_timestamp = a_max_counts.loc[a_max_counts.groupby(['source'])['timestamp'].idxmin()]

        # Define the final dataframe results with the peak of source needing observation in optimal
        # conditions, grouping by timestamp. Including the info of the number of sources where the
        # peak is found, for the observavility time window with optimal condition for those sources
        a_final_optimal_cond = a_timestamp.query('B3_soft_source_to_observe > 0').groupby(
            ['timestamp', 'B3_soft_source_to_observe']
        )[['source']].aggregate(lambda x: len(x.unique().tolist())).reset_index().rename(
            columns={'B3_soft_source_to_observe': 'peak_count', 'source': 'num_orig_source'})

        # Adding to the final dataframe results the sources available to be observed with SSR conditions
        # including an amp cal
        a_final_optimal_cond = a_final_optimal_cond.merge(a_ampcal_cond.groupby(
            ['timestamp']
        )[['source']].aggregate(lambda x: x.unique().tolist()).rename(
            columns={'source': 'source_list'}).reset_index().drop_duplicates(subset=['timestamp'], keep='last')
                                                          , on='timestamp', how='left')
        prim_ampcal = a.query('kind_b3 == 1 and timestamp in @a_final_optimal_cond.timestamp.tolist()'
                              ' and selSoftAll and Band3'
                              ).groupby(
            ['timestamp']
        )[['source']].aggregate(lambda x: x.unique().tolist()).reset_index().rename(
            columns={'source': 'prim_ampcal_list'}).drop_duplicates(subset=['timestamp'], keep='last')
        a_final_optimal_cond = a_final_optimal_cond.merge(prim_ampcal, on='timestamp', how='left')

        second_ampcal = a.query('kind_b3 == 2 and timestamp in @a_final_optimal_cond.timestamp.tolist()'
                                ' and selSoftAll'
                                ).groupby(
            ['timestamp']
        )[['source']].aggregate(lambda x: x.unique().tolist()).reset_index().rename(
            columns={'source': 'second_ampcal_list'}).drop_duplicates(subset=['timestamp'])
        a_final_optimal_cond = a_final_optimal_cond.merge(second_ampcal, on='timestamp', how='left')

        a_final_optimal_cond['num_obs'] = a_final_optimal_cond.source_list.apply(lambda x: len(x))
        a_final_optimal_cond['conditions'] = a_final_optimal_cond.timestamp.apply(
            lambda x: "optimal prim_ampcal" if x in prim_ampcal_list_timestamp else "optimal second_ampcal")
        # List of source included in  the observing plan
        list_to_observe = a_ampcal_cond.query(
            'timestamp in @a_final_optimal_cond.timestamp.tolist()').source.unique().tolist()
        print("Sources to observe: ", list_to_observe)
        # List of source not included in  the observing plan but with ampcal
        list_pending_to_observe = a_ampcal_cond.query('source not in @list_to_observe').source.unique().tolist()
        print("Sources pending to observe: ", list_pending_to_observe)
        # List of source without ampcal or out of the observavility time window defined
        list_non_ampcal = a_source_to_observe.query(
            'source not in @list_to_observe and source not in @list_pending_to_observe').source.unique().tolist()
        print("Source with non ampcal: ", list_non_ampcal)
        # List of source non observable
        list_non_observable = a.query(
            '(kind_b3 == 3  or kind_b3 == 4)'
            'and source not in @list_to_observe '
            'and source not in @list_pending_to_observe '
            'and source not in @list_non_ampcal').source.unique().tolist()
        print("Source non observable: ", list_non_observable)
        return a_final_optimal_cond

    def get_observation_windows(self, soft_const: bool = False, simulation: bool = False):

        """
        :param soft_const:
        """
        if simulation:
            a_frame = self.simulation_frame.copy()
        else:
            a_frame = self.main_frame.copy()
        if soft_const:
            find_windows = a_frame.query(
                'selSoftAll == True'
            ).groupby(
                ['source', 'timestamp']
            ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}).sort_values(by=['source', 'timestamp'])
        else:
            find_windows = a_frame.query(
                'selAll == True'
            ).groupby(
                ['source', 'timestamp']
            ).aggregate({'Band3': sum, 'Band6': sum, 'Band7': sum}).sort_values(by=['source', 'timestamp'])

        a = find_windows.reset_index().sort_values(by=['source', 'timestamp'])
        a = a.merge((find_windows - find_windows.shift(1)).reset_index(), on=['source', 'timestamp'],
                    suffixes=('', '_diff'))
        a = a.merge((find_windows - find_windows.shift(-1)).reset_index(), on=['source', 'timestamp'],
                    suffixes=('', '_diff_prev'))
        a['timediff'] = a.timestamp.diff().apply(lambda x: x.total_seconds() / 3600.0).fillna(-24.)
        a['timediff_prev'] = a.timestamp.diff(periods=-1).apply(lambda x: x.total_seconds() / 3600.0).fillna(24.)
        kind_source = a_frame[['source', 'kind_b3', 'kind_b6', 'kind_b7', 'kind_b9']].drop_duplicates()

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
            np.sin(np.deg2rad(lst * 15.) - np.deg2rad(ra)),
            np.cos(np.deg2rad(lst * 15.) - np.deg2rad(ra))
        )
    ) / 15.

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
