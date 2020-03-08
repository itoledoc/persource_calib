from tools import *
from astropy.coordinates.angles import Angle
from astropy.coordinates import AltAz
from astropy.utils import iers

iers.conf.auto_download = True
iers.conf.remote_timeout = 40


class FluxcalObs(object):

    def __init__(self, start, stop, step='15min'):

        self.epoch = {'start': start, 'stop': stop, 'step': step}

        self.sun = get_sso_coordinates('Sun', self.epoch)
        self.moon = get_sso_coordinates('Moon', self.epoch)
        self._io = get_sso_coordinates('Io', self.epoch)
        self._europa = get_sso_coordinates('Europa', self.epoch)
        self._ganymede = get_sso_coordinates('Ganymede', self.epoch)
        self._callisto = get_sso_coordinates('Callisto', self.epoch)
        self._jupiter = get_sso_coordinates('Jupiter', self.epoch)

        self.ganymede = self.create_source(
            'Ganymede', skycoord_dict={
                'Jupiter': self._jupiter, 'Io': self._io,
                'Europa': self._europa, 'Callisto': self._callisto,
                'Sun': self.sun, 'Moon': self.moon})
        self.callisto = self.create_source(
            'Callisto', skycoord_dict={
                'Jupiter': self._jupiter, 'Io': self._io,
                'Europa': self._europa, 'Ganymede': self._ganymede,
                'Sun': self.sun, 'Moon': self.moon})

        self._steps = len(self.sun.obstime)

        self.mars = self.create_source('Mars')
        self.uranus = self.create_source('Uranus')
        self.neptune = self.create_source('Neptune')

    def create_source(
            self, name: str, ra: Angle = None, dec: Angle = None,
            skycoord_dict: Dict[str, SkyCoord] = None
    ) -> pd.DataFrame:
        """

        :param name:
        :param ra:
        :param dec:
        :param sun:
        :param moon:
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
            df = self.build_dataframe(
                coords, skycoord_dict)
        else:
            df = self.build_dataframe(
                coords, {'Sun': self.sun, 'Moon': self.moon})

        self._prepare_coords_df(df, coords)

        return df

    @staticmethod
    def build_dataframe(coords, skycoord_dict):
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
    def _prepare_coords_df(df: pd.DataFrame, coords: SkyCoord) -> None:
        """
          Add extra column parameters to DataFrame produced by
        get_separations_dataframe.
          These parameters are Altitude, LST and HA, and they are under the columns
        `altitude`, `lst` and `ha`

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

    @staticmethod
    def cal_selector(
            df: pd.DataFrame, horizon: float = 20.,
            sun_limit: float = 21., moon_limit: float = 5.
    ) -> None:

        """

        :param horizon:
        :param sun_limit:
        :param moon_limit:
        :return:
        :param df:
        :return:
        """

        df['selAlt'] = df.altitude.apply(lambda x: x >= horizon)
        df['selSun'] = df.Sun.apply(lambda x: x >= sun_limit * 3600)
        df['selMoon'] = df.Moon.apply(lambda x: x >= moon_limit * 3600)

        band_columns = ['Band3', 'Band6', 'Band7','Band9']

        df[band_columns] = df.closest_distance.apply(
            lambda x: band_limits(x))
