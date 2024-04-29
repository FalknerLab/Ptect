from territorytools.urineV2 import Peetector


if __name__ == '__main__':
    avi_f = 'D:/TerritoryTools/tests/test_data/test_therm.avi'
    slp_f = 'D:/TerritoryTools/tests/test_data/test_therm_sleap.h5'
    pt = Peetector(avi_f, slp_f)
    # pt.add_dz()
    dz0 = pt.peetect_frames()
