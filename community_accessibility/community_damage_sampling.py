from .base import *

def load_inputs(inventory_filename, ground_motion_filename):
    """
    Loads the building inventory and ground motion data

    Parameters
    ----------
    inventory_filename: string
        name of a csv file with building inventory, including site lat, site lng, site Vs30

    ground_motion_filename: string
        name of an .h5 file containing the ground motions for each rupture and site


    """
    # retrieve the building inventory
    bldgs = pd.read_csv(inventory_filename)
    bldgs = bldgs.set_index('id')
    # add a column for the building period
    bldgs.insert(3, 'building.period', 0)
    bldgs.insert(4, 'building.period_retrofit', 0)

    # retrieve the ground motion data
    ruptures = pd.read_hdf(ground_motion_filename, 'Ruptures')
    gm_sites = pd.read_hdf(ground_motion_filename, 'Sites')
    with h5py.File(ground_motion_filename, 'r') as hf:
        ground_motion = hf['GroundMotions'][:]
        gm_periods = hf['Periods'][:]

    # confirm that the gm periods are sorted
    if not all(gm_periods[i] <= gm_periods[i + 1] for i in range(len(gm_periods) - 1)):
        raise ValueError('The period index of the ground motions is not sorted.')

    return bldgs, ruptures, gm_sites, gm_periods, ground_motion


def check_for_missing_vulnerabilities(bldgs, vulnerability_filename):
    """
    Raises an error if and buildings are missing vulnerability profiles

    Parameters
    ----------
    bldgs: pandas dataframe
        building inventory

    vulnerability_filename: string
        name of an .h5 file containing vulnerability profiles for each building id


    """
    # retrieve building ids
    bldg_ids = bldgs.index.values

    # retrieve vulnerability ids
    with h5py.File(vulnerability_filename, 'r') as hf:
        vulnerability_ids = list(hf.keys())

    # check for missing vulnerabilities
    missing_vulnerabilities = set(bldg_ids).difference(vulnerability_ids)
    if len(missing_vulnerabilities) > 0:
        print('The following buildings are missing vulnerability profiles:')
        print(missing_vulnerabilities)
        raise ValueError('Not all buildings are represented in the vulnerability profile')


def get_site_idx(bldg_id, bldgs, gm_sites):
    """
    Identifies the nearest ground motion site with the same soil Vs30

    Parameters
    ----------
    bldg_id: string
        index of the building of interest
    bldgs: pandas dataframe
        inventory of the buildings
    gm_sites: pandas dataframe
        list of ground motion sites with Vs30 values

    Returns
    -------
    site_idx: integer
        location index of the relevant site

    """
    # retrieve soil Vs30 and location for bldg
    bldg_vs30 = bldgs.loc[bldg_id, 'site.vs30']
    [bldg_x, bldg_y] = utm_conversion(bldgs.loc[bldg_id, 'site.lat'], bldgs.loc[bldg_id, 'site.lng'])

    # retrieve locations of the relevant sites with the same Vs30
    relevant_sites = gm_sites[gm_sites['Vs30'] == bldg_vs30]
    [sites_x, sites_y] = utm_conversion(relevant_sites['Latitude'], relevant_sites['Longitude'])

    # retrieve site idx of the nearest relevant soil type
    distances = ((sites_x - bldg_x) ** 2 + (sites_y - bldg_y) ** 2) ** 0.5
    nearest_idx = distances.argmin()
    site_id = relevant_sites.index.values[nearest_idx]
    site_idx = gm_sites.index.get_loc(site_id)

    return site_idx


def get_sa_values(ground_motion, gm_periods, site_idx, bldg_t, sim_idx):
    """
    Identifies the relevant Sa(T) for the building by interpolating from the response spectra for each scenario

    Parameters
    ----------
    ground_motion: numpy array
        contains the simulated ground motions [n_ruptures, n_sites, n_periods, n_realizations]

    gm_periods: numpy array
        contains the periods represented in the ground motions

    site_idx: int
        the index of the relevant site within the ground motions

    bldg_t: numpy array
        the value of the building period

    sim_idx: boolean
        index for sampling n_realizations from the n_sims ground motions

    Returns
    -------
    sa_t: numpy array
        the Sa(T) for the building [n_ruptures, n_realizations]

    pga: numpy array
        the PGA (approximated as Sa(T=0.01s) for the building [n_ruptures, n_realizations]

    """

    # identify the index of the periods above and below the building period
    t_above_idx = np.searchsorted(gm_periods, bldg_t, side='left')[0]
    t_above = gm_periods[t_above_idx]
    if t_above == 0:
        raise ValueError('bldg period, ' + str(bldg_t) + 'is less than the lowest ground motion period.')

    t_below_idx = t_above_idx - 1
    t_below = gm_periods[t_below_idx]

    # retrieve the relevant ruptures and realizations for the nearest periods
    sa_t_above = ground_motion[:, site_idx, t_above_idx, sim_idx]
    sa_t_below = ground_motion[:, site_idx, t_below_idx, sim_idx]

    # interpolate for Sa(T)
    d_t = (bldg_t - t_below) / (t_above - t_below)
    sa_t = sa_t_below + d_t * (sa_t_above - sa_t_below)

    # retrieve pga value (approximated as Sa(T=0.01s)
    pga_idx = np.where(gm_periods == 0.01)[0]
    pga = ground_motion[:, site_idx, pga_idx, sim_idx]

    return sa_t, pga


def sample_vulnerability(bldg_vulnerability, sa_t, sa_threshold, pga, pga_threshold, damage_threshold):
    """
    Samples building damage from the building vunerability, given the input Sa(T) values

    Parameters
    ----------
    bldg_vulnerability: hdf5 dset
        includes the vulnerability matrix [n_IM_levels, n_damage_parameters, n_simulations]
        attributes include the Sa(T) values representing each IM level

    sa_t: numpy array
        the target Sa(T) for the building [n_ruptures, n_realizations]

    sa_threshold: float or None
        the minimum Sa(T) for sampling damage

    pga: numpy array
        the PGA for the building [n_ruptures, n_realizations]

    pga_threshold: float or None
        the minimum PGA for sampling damage

    damage_threshold: float or None
        the minimum damage cost for recording repair times

    Returns
    -------
    bldg_damage: numpy array
        sampled damage for each rupture and realization [n_ruptures, n_damage_parameters, n_realizations]

    sampled_sa_t_indices: numpy array
        index of the IM levels for the sampling [n_ruptures, n_realizations]

    sampled_sa_t_value: numpy array
        the Sa(t) values for the sampling [n_ruptures, n_realizations]

     vulnerability_sampling_indices: numpy array
        index of the FEMA P-58 simulation for the sampling [n_ruptures, n_realizations]


    """

    # retrieve vulnerability metadata
    vuln_sa_t = bldg_vulnerability.attrs.__getitem__('saValues')
    [_, n_parameters, n_vuln_realizations] = bldg_vulnerability.shape
    bldg_vulnerability = bldg_vulnerability[:]

    # initialize outputs
    [n_rups, n_realizations] = sa_t.shape
    bldg_damage = np.full([n_rups, n_parameters, n_realizations], np.nan)
    sampled_sa_t_indices = np.full([n_rups, n_realizations], np.nan)
    sampled_sa_t_values = np.full([n_rups, n_realizations], np.nan)
    vulnerability_sampling_indices = np.full([n_rups, n_realizations], np.nan)

    # reset the sampling_thresholds if they are None
    if sa_threshold is None:
        sa_threshold = 0
    if pga_threshold is None:
        pga_threshold = 0

    # loop through ruptures and realizations for sampling indices
    for i_rup in range(n_rups):
        for i_real in range(n_realizations):

            # retrieve the pga for this realization
            i_pga = pga[i_rup, i_real]

            # retrieve the target sa_t for this realization
            target_sa_t = sa_t[i_rup, i_real]

            # if the PGA is too small, set the damage to zero
            if i_pga < pga_threshold:
                bldg_damage[i_rup, :, i_real] = 0
                sampled_sa_t_indices[i_rup, i_real] = 99
                sampled_sa_t_values[i_rup, i_real] = 0
                vulnerability_sampling_indices[i_rup, i_real] = np.nan
                # modify the idx if the target Sa(T) is also too small
                if target_sa_t < sa_threshold:
                    sampled_sa_t_indices[i_rup, i_real] = 9999

            # if the target Sa(T) is too small, set the damage to zero
            elif target_sa_t < sa_threshold:
                bldg_damage[i_rup, :, i_real] = 0
                sampled_sa_t_indices[i_rup, i_real] = 999
                sampled_sa_t_values[i_rup, i_real] = 0
                vulnerability_sampling_indices[i_rup, i_real] = np.nan

            # if Sa(T) is large enough, sample the damage
            else:
                # identify the nearest simulated Sa(T) value in for vulnerability
                vuln_sa_idx = (np.abs(vuln_sa_t - target_sa_t)).argmin()

                # generate a random sampling
                sampling_idx = np.random.randint(n_vuln_realizations)

                # store the sampling records
                sampled_sa_t_indices[i_rup, i_real] = vuln_sa_idx
                sampled_sa_t_values[i_rup, i_real] = vuln_sa_t[vuln_sa_idx]
                vulnerability_sampling_indices[i_rup, i_real] = sampling_idx

                # sample and store the damage parameters
                bldg_damage[i_rup, :, i_real] = bldg_vulnerability[vuln_sa_idx, :, sampling_idx]

                # if the functional repair time is zero, set max RCs = 0
                func_repair_idx = 2
                str_rc_idx = 4
                nonstr_rc_idx = 5
                if bldg_damage[i_rup, func_repair_idx, i_real] == 0:
                    bldg_damage[i_rup, [str_rc_idx, nonstr_rc_idx], i_real] = 0


            # if the repair cost is too low, consider the damage to be zero
            cost_idx = 11
            if bldg_damage[i_rup, cost_idx, i_real] < damage_threshold:
                bldg_damage[i_rup, :, i_real] = 0
                sampled_sa_t_indices[i_rup, i_real] = 99999


    return bldg_damage, sampled_sa_t_indices, sampled_sa_t_values, vulnerability_sampling_indices


def convert_occupancy_categories(df):
    """
    Converts the SP3 occupancy category ids from integers to words

    Parameters
    ----------
    df: pd dataframe
        inventory of the building inventory

    Returns
    -------
    df: pd dataframe
        same inventory with the occupancy categories defined

    """
    attribute = 'building.occupancy_id'
    sp3_occupancies = ['Commercial Office',
                       'Education',
                       'Education',
                       'Education',
                       'Healthcare',
                       'Hospitality',
                       'Residential',
                       'Research',
                       'Retail',
                       'Warehouse',
                       'Residential']
    for i in range(len(sp3_occupancies)):
        tag = i + 1
        df.loc[df[attribute] == tag, attribute] = sp3_occupancies[i]
        df[attribute] = df[attribute].fillna('Unknown')

    return df


def sample_community_damage(inventory_filename, ground_motion_filename, vulnerability_filename, output_filename, i_analysis, options):
    '''
    Combines building inventory, vulnerabilities, and ground motions to sample community damage

    Parameters
    ----------
    inventory_filename: string
        name of a .csv file containing the community building inventory

    ground_motion_filename: string
        name of an .h5 file containing data and metadata for ground motion maps

    vulnerability_filename: string
        name of an .h5 file containing vulnerability profiles corresponding to the building inventory

    output_filename: string
        name of the .h5 file that will store the data and metadata

    i_analysis: list of integers
        index of the folder in which to store the data

    options: dict
        includes key=n_realizations: int
            number of community damage maps to create
    Returns
    -------
    community_damage: np array
        damage metrics for each damage map [n_ruptures, n_bldgs, n_damage_parameters, n_realizations]

    '''
    # load the input data
    [bldgs, ruptures, gm_sites, gm_periods, ground_motion] = load_inputs(inventory_filename, ground_motion_filename)
    rupture_ids = ruptures.index.values
    bldg_ids = bldgs.index.values

    check_for_missing_vulnerabilities(bldgs, vulnerability_filename)

    # open vulnerability profiles
    vulnerability = h5py.File(vulnerability_filename, 'r')

    # determine size of output arrays
    n_ruptures = len(ruptures)
    n_bldgs = len(bldgs)
    [_, n_damage_parameters, _] = vulnerability[bldg_ids[0]].shape
    damage_parameters = vulnerability[bldg_ids[0]].attrs.__getitem__('parameters').split(',')

    # initialize outputs
    n_realizations = options['n_realizations']
    community_damage = np.full([n_ruptures, n_bldgs, n_damage_parameters, n_realizations], np.nan)
    site_indices = np.full(n_bldgs, np.nan, dtype='int64')
    gm_sa_t_values = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    gm_pga_values = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    sampled_sa_t_indices = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    sampled_sa_t_values = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    vulnerability_sampling_indices = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)

    # set the random index for ground motion map sampling
    [_,_,_,n_sims] = ground_motion.shape
    sim_where = np.random.choice(n_sims, n_realizations, replace=False)
    sim_idx = np.zeros(n_sims, dtype='bool')
    sim_idx[sim_where] = True

    for i_bldg in range(len(bldgs)):
        #         print('Bldg #' + str(i_bldg))
        # retrieve the relevant vulnerability profile and metadata
        bldg_vulnerability = vulnerability[bldg_ids[i_bldg]]
        bldg_t = bldg_vulnerability.attrs.__getitem__('Bldg_period')
        bldgs.loc[bldg_ids[i_bldg], 'building.period'] = bldg_t

        if np.isnan(bldgs.loc[bldg_ids[i_bldg], 'building.occupancy_id']):
            bldgs.loc[bldg_ids[i_bldg], 'building.occupancy_id'] = \
                bldg_vulnerability.attrs.__getitem__('Bldg_occupancy')[0]

        # identify the relevant ground motion site, per soil type and distance
        site_indices[i_bldg] = get_site_idx(bldg_ids[i_bldg], bldgs, gm_sites)

        # retrieve the Sa(T) values for the building for each ground motion realization
        [sa_t, pga] = get_sa_values(ground_motion, gm_periods, site_indices[i_bldg], bldg_t, sim_idx)
        gm_sa_t_values[:, i_bldg, :] = sa_t
        gm_pga_values[:, i_bldg, :] = pga

        # set a minimum Sa(T) threshold for damage to occur
        vuln_sa_t = bldg_vulnerability.attrs.__getitem__('saValues')
        sa_threshold = vuln_sa_t[0] / 2

        # set a minimum PGA threshold for damage to occur
        pga_threshold = 0.1

        # set a minimum damage repair cost threshold for keeping the sampled damage
        damage_threshold = 0.1

        # sample from the building vulnerability
        [community_damage[:, i_bldg, :, :],
         sampled_sa_t_indices[:, i_bldg, :],
         sampled_sa_t_values[:, i_bldg, :],
         vulnerability_sampling_indices[:, i_bldg, :]] = sample_vulnerability(bldg_vulnerability, sa_t, sa_threshold,
                                                                              pga, pga_threshold, damage_threshold)

    # close the vulnernability profiles
    vulnerability.close()

    # save community metadata to the output file
    # record building inventory
    bldgs = convert_occupancy_categories(bldgs)
    bldgs.to_hdf(output_filename, key='MetaData/buildings', mode='r+')

    # record the ruptures
    ruptures.to_hdf(output_filename, key='MetaData/ruptures', mode='r+')


    # save the community damage and underlying sampling results
    with h5py.File(output_filename, 'r+') as hf:

        # retrieve index of the output folder
        i_damage = i_analysis[0]

        # set up a new group for the results
        group_name = 'CommunityDamage_' + str(i_damage)
        group = hf['Results'].create_group(group_name)
        group.attrs['n_realizations'] = n_realizations


        # record community damage
        # damage_parameters = ['time_stable',
        #                      'time_reoccupancy',
        #                      'time_functional',
        #                      'time_full',
        #                      'max_structuralRC',
        #                      'max_nonstructuralRC',
        #                      'analytical_collapse',
        #                      'residual_drift',
        #                      'residual_collapse',
        #                      'isCollapse',
        #                      'replacementTriggered',
        #                      'repair_cost']

        dset = group.create_dataset('community_damage', data=community_damage)
        dset.attrs['n_realizations'] = n_realizations
        dset.attrs['n_ruptures'] = n_ruptures
        dset.attrs['n_bldgs'] = n_bldgs
        dset.attrs['damage_parameters'] = damage_parameters

        # record sampled hazard and vulnerability metadata
        return_periods = str.split(bldgs.loc[bldg_ids[0], 'hazard.return_periods'][1:-1], ',')
        return_periods = [int(i) for i in return_periods]

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/ground_motion_map_indices', data=sim_where)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/reference_site_indices', data=site_indices)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/ground_motion_SaT_values', data=gm_sa_t_values)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/ground_motion_pga_values', data=gm_pga_values)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/sampled_SaT_values', data=sampled_sa_t_values)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/sampled_hazard_indices',
                                    data=sampled_sa_t_indices)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/sampled_realizations',
                                    data=vulnerability_sampling_indices)

        # create subgroup for downtime logistics
        subgroup_name = 'DowntimeLogistics'
        group.create_group(subgroup_name)

    return community_damage


def sample_community_damage_with_retrofits(inventory_filename, ground_motion_filename, original_vulnerability_filename, retrofit_vulnerability_filename, output_filename, i_analysis, options):
    '''
    Combines building inventory, vulnerabilities, and ground motions to sample community damage

    Parameters
    ----------
    inventory_filename: string
        name of a .csv file containing the community building inventory

    ground_motion_filename: string
        name of an .h5 file containing data and metadata for ground motion maps

    original_vulnerability_filename: string
        name of an .h5 file containing vulnerability profiles corresponding to the as-is building inventory

    retrofit_vulnerability_filename: string
        name of an .h5 file containing vulnerability profiles corresponding to the retrofit building inventory

    output_filename: string
        name of the .h5 file that will store the data and metadata

    i_analysis: list of integers
        index of the folder in which to store the data

    options: dict
        includes key=n_realizations: int
            number of community damage maps to be created
    Returns
    -------
    community_damage: np array
        damage metrics for each damage map [n_ruptures, n_bldgs, n_damage_parameters, n_realizations]

    '''
    # load the input data
    [bldgs, ruptures, gm_sites, gm_periods, ground_motion] = load_inputs(inventory_filename, ground_motion_filename)
    rupture_ids = ruptures.index.values
    bldg_ids = bldgs.index.values

    check_for_missing_vulnerabilities(bldgs, original_vulnerability_filename)
    check_for_missing_vulnerabilities(bldgs, retrofit_vulnerability_filename)

    # open vulnerability profiles
    original_vulnerability = h5py.File(original_vulnerability_filename, 'r')
    retrofit_vulnerability = h5py.File(retrofit_vulnerability_filename, 'r')
    retrofit_status = options['retrofit_status']

    # determine size of output arrays
    n_ruptures = len(ruptures)
    n_bldgs = len(bldgs)
    [_, n_damage_parameters, _] = original_vulnerability[bldg_ids[0]].shape
    damage_parameters = original_vulnerability[bldg_ids[0]].attrs.__getitem__('parameters').split(',')

    # initialize outputs
    n_realizations = options['n_realizations']
    community_damage = np.full([n_ruptures, n_bldgs, n_damage_parameters, n_realizations], np.nan)
    site_indices = np.full(n_bldgs, np.nan, dtype='int64')
    gm_sa_t_values = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    gm_pga_values = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    sampled_sa_t_indices = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    sampled_sa_t_values = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)
    vulnerability_sampling_indices = np.full([n_ruptures, n_bldgs, n_realizations], np.nan)

    # set the random index for ground motion map sampling
    [_,_,_,n_sims] = ground_motion.shape
    sim_where = np.random.choice(n_sims, n_realizations, replace=False)
    sim_idx = np.zeros(n_sims, dtype='bool')
    sim_idx[sim_where] = True

    for i_bldg in range(len(bldgs)):
        #         print('Bldg #' + str(i_bldg))
        # retrieve the relevant vulnerability profile and metadata
        original_metadata = original_vulnerability[bldg_ids[i_bldg]]
        bldg_t = original_metadata.attrs.__getitem__('Bldg_period')
        bldgs.loc[bldg_ids[i_bldg], 'building.period'] = bldg_t

        if np.isnan(bldgs.loc[bldg_ids[i_bldg], 'building.occupancy_id']):
            bldgs.loc[bldg_ids[i_bldg], 'building.occupancy_id'] = \
                original_metadata.attrs.__getitem__('Bldg_occupancy')[0]

        retrofit_metadata = retrofit_vulnerability[bldg_ids[i_bldg]]
        bldg_t = retrofit_metadata.attrs.__getitem__('Bldg_period')
        bldgs.loc[bldg_ids[i_bldg], 'building.period_retrofit'] = bldg_t

        if retrofit_status.loc[bldg_ids[i_bldg],'status'] == 'retrofitted':
            bldg_vulnerability = retrofit_vulnerability[bldg_ids[i_bldg]]
            # print(bldg_ids[i_bldg] + ' retrofit date:')
            # print(bldg_vulnerability.attrs.__getitem__('Bldg_date')[0])
        else:
            bldg_vulnerability = original_vulnerability[bldg_ids[i_bldg]]
            # print(bldg_ids[i_bldg] + ' original date:')
            # print(bldg_vulnerability.attrs.__getitem__('Bldg_date')[0])
        bldg_t = bldg_vulnerability.attrs.__getitem__('Bldg_period')

        # identify the relevant ground motion site, per soil type and distance
        site_indices[i_bldg] = get_site_idx(bldg_ids[i_bldg], bldgs, gm_sites)

        # retrieve the Sa(T) values for the building for each ground motion realization
        [sa_t, pga] = get_sa_values(ground_motion, gm_periods, site_indices[i_bldg], bldg_t, sim_idx)
        gm_sa_t_values[:, i_bldg, :] = sa_t
        gm_pga_values[:, i_bldg, :] = pga

        # set a minimum Sa(T) threshold for damage to occur
        vuln_sa_t = bldg_vulnerability.attrs.__getitem__('saValues')
        sa_threshold = vuln_sa_t[0] / 2

        # set a minimum PGA threshold for damage to occur
        pga_threshold = 0.1

        # set a minimum damage repair cost threshold for keeping the sampled damage
        damage_threshold = 0.1

        # sample from the building vulnerability
        [community_damage[:, i_bldg, :, :],
         sampled_sa_t_indices[:, i_bldg, :],
         sampled_sa_t_values[:, i_bldg, :],
         vulnerability_sampling_indices[:, i_bldg, :]] = sample_vulnerability(bldg_vulnerability, sa_t, sa_threshold,
                                                                              pga, pga_threshold, damage_threshold)

    # close the vulnernability profiles
    original_vulnerability.close()
    retrofit_vulnerability.close()

    # save community metadata to the output file
    # record building inventory
    bldgs = convert_occupancy_categories(bldgs)
    bldgs.to_hdf(output_filename, key='MetaData/buildings', mode='r+')

    # record the ruptures
    ruptures.to_hdf(output_filename, key='MetaData/ruptures', mode='r+')


    # save the community damage and underlying sampling results
    with h5py.File(output_filename, 'r+') as hf:

        # retrieve index of the output folder
        i_damage = i_analysis[0]

        # set up a new group for the results
        group_name = 'CommunityDamage_' + str(i_damage)
        group = hf['Results'].create_group(group_name)
        group.attrs['n_realizations'] = n_realizations

        # record community damage
        # damage_parameters = ['time_stable',
        #                      'time_reoccupancy',
        #                      'time_functional',
        #                      'time_full',
        #                      'max_structuralRC',
        #                      'max_nonstructuralRC',
        #                      'analytical_collapse',
        #                      'residual_drift',
        #                      'residual_collapse',
        #                      'isCollapse',
        #                      'replacementTriggered',
        #                      'repair_cost']

        dset = group.create_dataset('community_damage', data=community_damage)
        dset.attrs['n_realizations'] = n_realizations
        dset.attrs['n_ruptures'] = n_ruptures
        dset.attrs['n_bldgs'] = n_bldgs
        dset.attrs['damage_parameters'] = damage_parameters

        # record sampled hazard and vulnerability metadata
        return_periods = str.split(bldgs.loc[bldg_ids[0], 'hazard.return_periods'][1:-1], ',')
        return_periods = [int(i) for i in return_periods]

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/ground_motion_map_indices', data=sim_where)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/reference_site_indices', data=site_indices)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/ground_motion_SaT_values', data=gm_sa_t_values)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/ground_motion_pga_values', data=gm_pga_values)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/sampled_SaT_values', data=sampled_sa_t_values)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/sampled_hazard_indices',
                                    data=sampled_sa_t_indices)

        dset = group.create_dataset('Hazard_and_Vulnerability_Sampling/sampled_realizations',
                                    data=vulnerability_sampling_indices)

        # create subgroup for downtime logistics
        subgroup_name = 'DowntimeLogistics'
        group.create_group(subgroup_name)

        # record retrofit status
        key = 'Results/' + group_name + '/retrofit_status'
        retrofit_status.to_hdf(output_filename, key=key, mode='r+')

    return community_damage