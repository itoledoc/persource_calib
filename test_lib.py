'''
Documentation
'''

import pytest
from fluxcal_observer import *


test_source = SkyCoord(ra=10.2 * u.hour, dec=-22.1*u.deg, frame='icrs')

epoch1 = {'start': '2020-03-01', 'stop': '2020-03-03', 'step': '15min'}
epoch2 = {'start': '2020-03-02', 'stop': '2020-03-04', 'step': '15min'}
epoch3 = {'start': '2020-03-03', 'stop': '2020-03-04', 'step': '15min'}


def test_ha():
    assert calc_ha(12.0 * 15., 12.0) == 0
    assert calc_ha(0, 12.0) == 12


def test_get_sso_coordinates():

    sun = get_sso_coordinates('Sun', epoch=epoch1)
    moon = get_sso_coordinates('Moon', epoch=epoch1, raw_table=True)

    assert type(sun) == SkyCoord
    assert type(moon) == Table
