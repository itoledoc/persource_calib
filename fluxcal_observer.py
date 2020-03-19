from tools import *
from astropy.coordinates.angles import Angle
from astropy.coordinates import AltAz
from astropy.utils import iers

iers.conf.auto_download = True
iers.conf.remote_timeout = 40


class FluxcalObs:

    def __init__(self, start: str, stop: str, step: str = '15min'):
        """
          Class to handle the observability of Calibrators that need monitoring.

        :param start: a start date in format 'YYYY-MM-DD', as string.
        :param stop: an end date, string. Same format
        :param step: optional, every how much time do we information (specially
          relevant for ephemeris objects)

        Retunrs an instance

        """
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
        """
          Auxiliary method to create sources given a name, coordinates, and a
        dictionary. The dictionary contains SkyCoord objects for the reference
        sources that will be used to calculate limiting distances.

        :param name:
        :param ra:
        :param dec:
        :param skycoord_dict:
        :param debug:
        :return:
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
