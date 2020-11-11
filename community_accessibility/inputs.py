from .base import *
from .mapping import *



def load_community_realizations(filename, show, view_map):
    """

    Parameters
    ----------
    filename : str
        name of an .h5 file containing the realizations of the community Monte Carlo Analysis
    show : bool
        True returns a printed description of the file structure
    view_map : bool
        True returns a map of the building inventory based on occupancy and structural type

    Returns
    -------

    """

    # read the pandas dataframes
    bldgs = pd.read_hdf(filename, 'MetaData/buildings')
    ruptures = pd.read_hdf(filename, 'MetaData/ruptures')

    # read the realizations data
    with h5py.File(filename, 'r') as hdf:
        dset = hdf['CommunityResults/downtime']
        downtime_parameters = [x.decode('ascii') for x in dset.attrs.__getitem__('downtime_parameters')]
        impeding_factor_paths = [x.decode('ascii') for x in dset.attrs.__getitem__('impeding_factor_paths')]
        results = dset[:]

        #########
        # temporarily add the residual drift from the damages
        res_drift = 7
        dset = hdf['CommunityResults/damage']
        [_, _, _, n] = results.shape
        results = np.concatenate((results, dset[:, :, :, res_drift][:, :, :, None]), axis=3)
        downtime_parameters.append('residualDrift')
        # there was an error in saving the downtime parameters, hard coded below:
        downtime_parameters = ['inspection', 'engineering_mobilization', 'financing', 'contractor_mobilization',
                               'permiting', 'engineering_path', 'financing_path', 'contractor_path', 'controlling_path',
                               'impeding_factor_delay', 'stable_repair', 'stable_downtime', 'functional_repair',
                               'functional_downtime', 'residualDrift']
        # later this should be handled during the original analysis/storage
        #########

    if show:
        with h5py.File(filename, 'r') as hdf:
            group_keys = list(hdf.keys())
            print('The hdf groups include:')
            print(group_keys)
            print()

            for g in group_keys:
                print()
                group = hdf.get(g)
                dset_keys = list(group.keys())
                print('Datasets in the ' + g + ' group:')
                print(dset_keys)
                print()

                if g != 'MetaData':
                    for k in dset_keys:
                        dset = group[k]
                        print('Attributes in the ' + k + ' dataset:')
                        print(list(dset.attrs.keys()))
                        print('Shape of dataset is: ' + str(dset.shape))
                        print()

    if view_map:
        map_buildings(bldgs, attribute='building.building_type_id', color_breaks='match')
        map_buildings(bldgs, attribute='building.occupancy_id', color_breaks='match')

    return bldgs, ruptures, results, downtime_parameters, impeding_factor_paths