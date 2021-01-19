from .base import *
from .impeding_factor_parameter_identification import  *
from .impeding_factor_assigments import *
from .mapping import *


def simulate_impeding_factors(mitigation, if_dictionary, n_sims, n_rups, n_bldgs, output_filename):
    # assign distribution parameters based on the tags
    medians = np.full([len(mitigation), 9], np.nan)
    betas = np.full([len(mitigation), 9], np.nan)
    parameters = list(mitigation.keys())[-5:]
    parameter_idx = [0, [1, 2, 3], 4, [5, 6], [7, 8]]

    for p, i in zip(parameters, parameter_idx):
        [medians[:, i], betas[:, i]] = eval(p + '_parameters(mitigation, if_dictionary)')

    # generate impeding factors based on the distribution
    if_pool = np.full([n_rups, n_bldgs, 9, n_sims], np.nan)
    for i in range(n_sims):
        for j in range(n_rups):
            if_pool[j, :, :, i] = np.random.lognormal(np.log(medians), betas)

    return medians, betas, if_pool


def sample_impeding_factors(bldgs, community_damage, output_filename, i_analysis, options):
    n_values = 14

    # associate mitigation options with each building
    mitigation = options['mitigation']
    mitigation = pd.concat([bldgs, mitigation], axis=1)
    if_parameters = list(mitigation.keys())[-5:]

    [n_rups, n_bldgs, _, n_sims] = community_damage.shape
    impeding_factors = np.zeros([n_rups, n_bldgs, n_values, n_sims])

    # load impeding factors parameter dictionary
    with open('input scripts\ImpedingFactors_ParameterDictionary.json', 'r') as inFile:
        if_dictionary = json.load(inFile)

    # simulate all the possible impeding factors
    [medians, betas, simulated_if_pool] = simulate_impeding_factors(mitigation, if_dictionary, n_sims, n_rups, n_bldgs, None)
    if_pool = simulated_if_pool
    [_, _, n_ifs, _] = if_pool.shape

    # weight all impeding factors for building specific sensitivity studies
    if_bldg_weights = options['impeding_factor_building_weights']
    # expand to fit the shape of the impeding factors
    if_bldg_weights = np.repeat(np.expand_dims(if_bldg_weights, 0), n_rups, axis=0)
    if_bldg_weights = np.repeat(np.expand_dims(if_bldg_weights, 3), n_sims, axis=3)
    if_pool = if_bldg_weights * if_pool

    # weight all impeding factors for total community damage sensitivity studies
    key = 'community_damage_weights'
    if key in options.keys():
        community_damage_parameter = options[key]['community_damage_parameter']
        community_damage_threshold = options[key]['community_damage_threshold']
        community_damage_percent = np.mean(np.squeeze(community_damage[:, :, community_damage_parameter, :]) >= community_damage_threshold, axis=1)

        min_community_damage = options[key]['min_community_damage']
        max_community_damage = options[key]['max_community_damage']
        min_damage_weight = options[key]['min_damage_weight']
        max_damage_weight = options[key]['max_damage_weight']

        if_community_damage_weights = np.interp(community_damage_percent,
                                                [min_community_damage, max_community_damage],
                                                [min_damage_weight, max_damage_weight])

        # expand to fit the shape of the impeding factors (with a new name so the original 2D matrix can be saved below)
        if_community_weights = if_community_damage_weights
        if_community_weights = np.repeat(np.expand_dims(if_community_weights, 1), n_bldgs, axis=1)
        if_community_weights = np.repeat(np.expand_dims(if_community_weights, 2), n_ifs, axis=2)
        if_pool = if_community_weights * if_pool

    else:
        if_community_damage_weights = np.ones([n_rups, n_sims])

    # query the appropriate impeding factors, depending on the amount of damage
    for p, i in zip(if_parameters, range(5)):
        impeding_factors[:, :, i, :] = eval(p + '_time(community_damage, if_pool)')

    # combine the impeding factors to get the parallel paths for REDi
    path_component_idx = [[0, 1, 4], [0, 2], [0, 3]]
    ## LONG-LEAD TIME IS IGNORED, as in SP3 ##
    path_idx = [5, 6, 7]
    for c, i in zip(path_component_idx, path_idx):
        # add each relevant impeding factor to sum the paths
        for idx in c:
            impeding_factors[:, :, i, :] = impeding_factors[:, :, i, :] + impeding_factors[:, :, idx, :]

    # identify the controlling path
    tag = np.full([n_rups, n_bldgs, n_sims], np.nan)
    max_path = np.full([n_rups, n_bldgs, n_sims], np.nan)
    paths = impeding_factors[:, :, path_idx, :]
    max_idx = np.argmax(paths, axis=2)
    for p in range(3):
        idx = np.where(max_idx == p)
        tag[idx] = p
        max_path[idx] = paths[:, :, p, :][idx]

    impeding_factors[:, :, 8, :] = tag
    impeding_factors[:, :, 9, :] = max_path

    # include repair times and downtimes to stable recovery
    impeding_factors[:, :, 10, :] = community_damage[:, :, 0, :]
    impeding_factors[:, :, 11, :] = max_path + impeding_factors[:, :, 10, :]
    # include repair times and downtimes to functional recovery
    impeding_factors[:, :, 12, :] = community_damage[:, :, 2, :]
    impeding_factors[:, :, 13, :] = max_path + impeding_factors[:, :, 12, :]

    # save the impeding factors and underlying sampling results
    with h5py.File(output_filename, 'r+') as hf:

        # retrieve index of the output folder
        i_damage = i_analysis[0]
        i_impeding_factors = i_analysis[1]

        # set up a new group for the results
        group_name = 'CommunityDamage_' + str(i_damage) +\
                     '/DowntimeLogistics/ImpedingFactors_' + str(i_impeding_factors)
        group = hf['Results'].create_group(group_name)
        group.attrs['impeding_factor_parameter_dictionary'] = str(if_dictionary)

        # record impeding factors
        downtime_parameters = ['inspection',
                               'engineering_mobilization',
                               'financing',
                               'contractor_mobilization',
                               'permiting',
                               'engineering_path',
                               'financing_path',
                               'contractor_path',
                               'controlling_path',
                               'impeding_factor_delay',
                               'stable_repair',
                               'stable_downtime',
                               'functional_repair',
                               'functional_downtime']
        impeding_paths = ['engineering', 'financing', 'contractor']
        impeding_paths_idx = [0, 1, 2]

        dset = group.create_dataset('impeding_factors', data=impeding_factors)
        dset.attrs['n_sims'] = n_sims
        dset.attrs['n_rups'] = n_rups
        dset.attrs['n_bldgs'] = n_bldgs
        dset.attrs['downtime_parameters'] = downtime_parameters
        dset.attrs['impeding_factor_paths'] = impeding_paths
        dset.attrs['impeding_paths_idx'] = impeding_paths_idx

        # record sampled impeding factor metadata
        impeding_factor_names = ['Inspection',
                                 'EngMob_Low',
                                 'EngMob_High',
                                 'Eng_Mob_Repl',
                                 'Financing',
                                 'ContrMob_Low',
                                 'ContrMob_High',
                                 'Permit_Low',
                                 'Permit_High']
        impeding_factor_names = impeding_factor_names

        subgroup_name = 'ImpedingFactorSampling'
        subgroup = group.create_group(subgroup_name)
        subgroup.attrs['impeding_factor_names'] = impeding_factor_names

        dset = subgroup.create_dataset('medians', data=medians)
        dset = subgroup.create_dataset('betas', data=betas)
        dset = subgroup.create_dataset('simulated_impeding_factor_pool', data=simulated_if_pool)
        dset = subgroup.create_dataset('weighted_impeding_factor_pool', data=if_pool)
        dset = subgroup.create_dataset('impeding_factor_building_weights', data=options['impeding_factor_building_weights'])
        dset = subgroup.create_dataset('impeding_factor_community_damage_weights', data=if_community_damage_weights)

        if 'community_damage_weights' in options.keys():
            dset.attrs['community_damage_parameter'] = options['community_damage_weights']['community_damage_parameter']
            dset.attrs['community_damage_threshold'] = options['community_damage_weights']['community_damage_threshold']
            dset.attrs['min_community_damage'] = options['community_damage_weights']['min_community_damage']
            dset.attrs['max_community_damage'] = options['community_damage_weights']['max_community_damage']
            dset.attrs['min_damage_weight'] = options['community_damage_weights']['min_damage_weight']
            dset.attrs['max_damage_weight'] = options['community_damage_weights']['max_damage_weight']
            dset.attrs['community_damage_percent'] = community_damage_percent
            dset.attrs['average_community_damage_weight_per_rupture'] = np.mean(if_community_damage_weights, axis=1)


        # create subgroup for cordon logistics
        subgroup_name = 'CordonLogistics'
        group.create_group(subgroup_name)

    # add the mitigation strategies for each building
    pd_name = 'Results/CommunityDamage_' + str(i_damage) + \
              '/DowntimeLogistics/ImpedingFactors_' + str(i_impeding_factors) + \
              '/ImpedingFactorSampling/mitigation_strategies'

    mitigation.to_hdf(output_filename, key=pd_name, mode='r+')

    return impeding_factors


def identify_cordon_locations(bldgs, height_threshold, radius_scale_factor):
    # identify the tall building and retrieve their heights
    df = bldgs
    hgt_col = 'building.building_ht_ft'
    df_tall = df[df[hgt_col] >= height_threshold]
    height = df_tall[hgt_col].values
    tall_idx = np.where(df[hgt_col] >= height_threshold)[0]

    # set the cordon radius (in meters)
    r = (radius_scale_factor / 3.28) * height

    # create a geojson of the cordons
    cordon_locations = df_tall[['site.lat', 'site.lng']].copy()
    cordon_locations['radius'] = r

    # convert WGS84 coordinates to meters
    [x_tall, y_tall] = utm_conversion(df_tall['site.lat'], df_tall['site.lng'])
    [x, y] = utm_conversion(df['site.lat'], df['site.lng'])

    # create directed adjancency matrix of potential cordon relationships
    cordon_adjacency = np.zeros([len(df), len(df_tall)])
        # tall buildings are columns
        # all buildings are rows
        # Aij = 1 if tall building, j, would cast a cordon over building i

    # cycle through each tall building to identify those that it influences
    for x_i, y_i, r_i, i in zip(x_tall, y_tall, r, range(sum(tall_idx))):
        # get the index of any building that is within the radius distance
        idx = np.where(np.sqrt(np.square(x - x_i) + np.square(y - y_i)) <= r_i)[0]
        cordon_adjacency[idx, i] = 1
        # remove the building causing the cordon as an affected building
        cordon_adjacency[tall_idx[i], i] = 0

    cordon_adjacency = pd.DataFrame(cordon_adjacency, index=df.index.values, columns=df_tall.index.values)

    return cordon_locations, cordon_adjacency


# def identify_cordon_durations(bldgs, cordon_locations, community_damage, impeding_factors, options):
#     damage_indicator_idx = options['cordon_damage_indicator_idx']
#     damage_threshold = options['cordon_trigger']
#     cordon_duration_parameter = options['cordon_duration_parameter']
#
#     # determine the size of the output matrix
#     [n_rups, _, _, n_sims] = community_damage.shape
#     n_tall = len(cordon_locations)
#     cordon_durations = np.zeros([n_rups, n_tall, n_sims])
#
#     # find the index of the tall buildings within the damage results
#     tall_idx = cordon_locations.index.values
#     bldg_idx = np.where(bldgs.index.isin(tall_idx))[0]
#
#     # retrieve the index of the buildings with cordons triggered by damage
#     community_damage = community_damage[:, bldg_idx, :, :]
#
#     # if damage indicator is not repair cost, include replacement cases where the damage parameters was set to zero
#     if damage_indicator_idx != 11:
#         replacement_indicator = 10 # replacement triggered (may be due to long repair times, not collapse)
#         [rup_dim, bldg_dim, sim_dim] = np.where(community_damage[:, :, replacement_indicator, :] == 1)
#         community_damage[rup_dim, bldg_dim, damage_indicator_idx, sim_dim] = damage_threshold
#
#     # retrieve the cordon durations based on the downtime to stabilization (without secondary cordon effects)
#     impeding_factors = impeding_factors[:, bldg_idx, :, :]
#
#     if cordon_duration_parameter == 'stable downtime':
#         cordon_duration_idx = 11  # recovery time to stabilitization
#     elif cordon_duration_parameter == 'stable repair':
#         replacement_stabilization_time = options['replacement_stabilization_time']
#         cordon_duration_idx = 10  # repair time to stabilitization
#         replacement_indicator = 10  # replacement triggered (may be due to long repair times, not collapse)
#         [rup_dim, bldg_dim, sim_dim] = np.where(community_damage[:, :, replacement_indicator, :] == 1)
#         impeding_factors[rup_dim, bldg_dim, cordon_duration_idx, sim_dim] = replacement_stabilization_time
#     else:
#         raise Warning('specify the parameter for the cordon duration')
#
#     [rup_dim, bldg_dim, sim_dim] = np.where(community_damage[:, :, damage_indicator_idx, :] >= damage_threshold)
#     cordon_durations[rup_dim, bldg_dim, sim_dim] = impeding_factors[rup_dim, bldg_dim, cordon_duration_idx, sim_dim]
#
#     return cordon_durations

def compare(a, op, b):

    ops = {"==": operator.eq,
           "!=": operator.ne,
           "<": operator.lt,
           "<=": operator.le,
           ">": operator.gt,
           ">=": operator.ge}

    return ops[op](a, b)

def identify_cordon_durations_by_trigger_condition(bldgs, cordon_locations, community_damage, impeding_factors, options):

    # set the stabilization time for each building, depending on the parameter
    cordon_duration_parameter = options['cordon_duration_parameter']
    if cordon_duration_parameter == 'stable downtime':
        cordon_duration_idx = 11  # recovery time to stabilitization
    elif cordon_duration_parameter == 'stable repair':
        replacement_stabilization_time = options['replacement_stabilization_time']
        cordon_duration_idx = 10  # repair time to stabilitization
        # if building will be replaced, update the stabilization from 0 days to the replacement stabilization time
        replacement_indicator = 10  # replacement triggered (may be due to long repair times, not collapse)
        [rup_dim, bldg_dim, sim_dim] = np.where(community_damage[:, :, replacement_indicator, :] == 1)
        impeding_factors[rup_dim, bldg_dim, cordon_duration_idx, sim_dim] = replacement_stabilization_time
    else:
        raise Warning('specify the parameter for the cordon duration')


    # determine the size of the output matrix
    [n_rups, _, _, n_sims] = community_damage.shape
    n_tall = len(cordon_locations)
    cordon_durations = np.zeros([n_rups, n_tall, n_sims])

    # find the index of buildings that may be cordoned
    tall_idx = cordon_locations.index.values

    cordon_trigger_parameters = options['cordon_trigger_parameters']
    n_conditions = len(cordon_trigger_parameters['trigger_condition_parameter'])
    for i_condition in range(n_conditions):
        condition_parameter = cordon_trigger_parameters['trigger_condition_parameter'][i_condition]
        condition_value = cordon_trigger_parameters['trigger_condition_value'][i_condition]
        condition_logical = cordon_trigger_parameters['trigger_condition_logical'][i_condition]

        # get the tall buildings that meet the inventory condition
        bldg_list = compare(bldgs[condition_parameter][tall_idx], condition_logical, condition_value)

        # if the condition is based on year, correct for any retrofit buildings
        if (condition_parameter == 'building.year_of_construction'):
            retrofit_status = options['retrofit_status'].loc[cordon_locations.index.values]
            retrofit = retrofit_status['status'] == 'retrofitted'
            if condition_logical == '<':
                bldg_list[retrofit] = False
            elif condition_logical == '>=':
                bldg_list = bldg_list | retrofit

        # get the index of the buildings within the full inventory
        bldg_idx = np.where(bldgs.index.isin(bldg_list.index[bldg_list]))[0]
        # get the index of the buildings within the list of possible cordonds
        cordon_idx = np.where(cordon_locations.index.isin(bldg_list.index[bldg_list]))[0]

        # get the trigger parameter and threshold
        cordon_damage_indicator = cordon_trigger_parameters['cordon_damage_indicator'][i_condition]
        damage_threshold = cordon_trigger_parameters['cordon_damage_threshold'][i_condition]
        temp_community_damage = community_damage.copy()
        if cordon_damage_indicator == 'max_sdr':
            damage_indicator_idx = 12
            # include collapse cases as sdr = threshold
            collapse_indicator = 6  # analytical collapse
            [rup_dim, bldg_dim, sim_dim] = np.where(community_damage[:, :, collapse_indicator, :] == 1)
            temp_community_damage[rup_dim, bldg_dim, damage_indicator_idx, sim_dim] = damage_threshold
        elif cordon_damage_indicator == 'analytical collapse':
            damage_indicator_idx = 6 # analytical collapse


        if sum(bldg_list) > 0:
            # get the indices for the buildings that trigger a cordon for the current condition
            [rup_dim, bldg_dim, sim_dim] = np.where(
                temp_community_damage[:, bldg_idx, damage_indicator_idx, :] >= damage_threshold)
            # assign the cordon durations
            cordon_durations[rup_dim, cordon_idx[bldg_dim], sim_dim] = impeding_factors[
                rup_dim, bldg_idx[bldg_dim], cordon_duration_idx, sim_dim]

    return cordon_durations


def propogate_cordons(bldgs, impeding_factors, cordon_adjacency, cordon_locations, cordon_durations):
    # specify new column names
    imp_col_names = ['inspection', 'engineering_mobilization', 'financing',
                     'contractor_mobilization', 'permitting', 'engineering_path',
                     'financing_path', 'contractor_path', 'controlling_path',
                     'impeding_factor_delay', 'stable_repair', 'stable_downtime',
                     'functional_repair', 'functional_downtime']
    cordon_col_names = ['n_cordons_present', 'cordon_duration', 'cordon_controls',
                        'cordon_induced_delay', 'total_delay', 'total_downtime']
    col_names = imp_col_names + cordon_col_names

    imp_delay_idx = col_names.index('impeding_factor_delay')
    func_repair_idx = col_names.index('functional_repair')
    n_cordons_idx = col_names.index('n_cordons_present')
    cordon_duration_idx = col_names.index('cordon_duration')
    cordon_control_idx = col_names.index('cordon_controls')
    cordon_delay_idx = col_names.index('cordon_induced_delay')
    delay_idx = col_names.index('total_delay')
    downtime_idx = col_names.index('total_downtime')

    # prepare results matrix
    [n_rups, n_bldgs, _, n_sims] = impeding_factors.shape
    [_, n_tall] = cordon_adjacency.shape
    community_downtime = np.concatenate((impeding_factors, np.zeros([n_rups, n_bldgs, len(cordon_col_names), n_sims])),
                                        axis=2)

    # expand the cordon durations and adjacency to get a weighted directed adjencency for each realizations
    cordon_durations_expanded = np.expand_dims(cordon_durations, axis=1)
    cordon_durations_expanded = np.tile(cordon_durations_expanded, (1, n_bldgs, 1, 1))

    cordon_adjacency_expanded = np.expand_dims(cordon_adjacency.values, axis=0)
    cordon_adjacency_expanded = np.expand_dims(cordon_adjacency_expanded, axis=3)
    cordon_adjacency_expanded = np.tile(cordon_adjacency_expanded, (n_rups, 1, 1, n_sims))

    cordon_duration_adjacency = np.multiply(cordon_durations_expanded, cordon_adjacency_expanded)

    ########
    # remove any externally imposed cordons from a building that requires a cordon
    # this prioritizes access to the critical buildings that impose cordons

    # find the index of the tall buildings within the community results
    tall_idx = cordon_locations.index.values
    bldg_idx = np.where(bldgs.index.isin(tall_idx))[0]

    # set critical building in each realization as not having an imposed cordon
    for i_rup in range(n_rups):
        for i_sim in range(n_sims):
            cordon_idx = cordon_durations[i_rup, :, i_sim] > 0
            cordon_duration_adjacency[i_rup, bldg_idx[cordon_idx], :, i_sim] = 0
    ########

    # find the number of cordons that affect each building
    community_downtime[:, :, n_cordons_idx, :] = np.sum(cordon_duration_adjacency > 0, axis=2)

    # find the controlling duration for each building
    community_downtime[:, :, cordon_duration_idx, :] = np.max(cordon_duration_adjacency, axis=2)

    # identify instances where the cordon controls over other impeding factors
    [rup_dim, bldg_dim, sim_dim] = np.where(
        community_downtime[:, :, cordon_duration_idx, :] > community_downtime[:, :, imp_delay_idx, :])
    community_downtime[rup_dim, bldg_dim, cordon_control_idx, sim_dim] = 1
    community_downtime[:, :, cordon_delay_idx, :] = community_downtime[:, :, cordon_duration_idx,
                                                    :] - community_downtime[:, :, imp_delay_idx, :]
    community_downtime[:, :, delay_idx, :] = np.maximum(community_downtime[:, :, cordon_duration_idx, :],
                                                        community_downtime[:, :, imp_delay_idx, :])
    community_downtime[:, :, downtime_idx, :] = community_downtime[:, :, delay_idx, :] + community_downtime[:, :,
                                                                                         func_repair_idx, :]

    return cordon_duration_adjacency, community_downtime, col_names


def evaluate_cordons(bldgs, community_damage, impeding_factors, output_filename, i_analysis, options):
    height_threshold = options['height_threshold']
    radius_scale_factor = options['radius_scale_factor']

    [cordon_locations, cordon_adjacency] = identify_cordon_locations(bldgs, height_threshold, radius_scale_factor)

    cordon_durations = identify_cordon_durations_by_trigger_condition(bldgs, cordon_locations, community_damage, impeding_factors,
                                                 options)

    [cordon_duration_adjacency, community_downtime, downtime_parameters] = propogate_cordons(bldgs, impeding_factors,
                                                                                   cordon_adjacency, cordon_locations,
                                                                                   cordon_durations)

    dt = options['dt']
    occ_labels = options['occ_labels']
    [community_recovery, time, n_bldgs, sqft_totals] = parallel_community_recovery(bldgs, community_downtime, downtime_parameters, occ_labels, dt)

    # save cordon logistics and underlying results
    with h5py.File(output_filename, 'r+') as hf:
        # retrieve index of the output folder
        i_damage = i_analysis[0]
        i_impeding_factors = i_analysis[1]
        i_cordons = i_analysis[2]

        # set up a new group for the results
        group_name = 'Results' + \
                     '/CommunityDamage_' + str(i_damage) + \
                     '/DowntimeLogistics/ImpedingFactors_' + str(i_impeding_factors) + \
                     '/CordonLogistics/Cordons_' + str(i_cordons)
        group = hf.create_group(group_name)
        group.attrs['height_threshold'] = height_threshold
        group.attrs['radius_scale_factor'] = radius_scale_factor

        # if damage_indicator_idx == 11:
        #     group.attrs['damage_indicator_name'] = 'repair cost'
        # elif damage_indicator_idx == 12:
        #     group.attrs['damage_indicator_name'] = 'maximum story drift ratio'
        # elif damage_indicator_idx == 7:
        #     group.attrs['damage_indicator_name'] = 'residual drift'
        # elif damage_indicator_idx == 9:
        #     group.attrs['damage_indicator_name'] = 'irreparable (analytical or residual drift)'
        # elif damage_indicator_idx == 8:
        #     group.attrs['damage_indicator_name'] = 'irreparable due to residual drift'
        # elif damage_indicator_idx == 6:
        #     group.attrs['damage_indicator_name'] = 'irreparable due to analytical collapse'
        # elif damage_indicator_idx == 0:
        #     group.attrs['damage_indicator_name'] = 'stability repair time'
        # else:
        #     raise Warning('specify the name of the damage indicator')
        # group.attrs['damage_indicator_idx'] = damage_indicator_idx
        # group.attrs['damage_threshold'] = damage_threshold
        group.attrs['cordon_trigger_parameters'] = json.dumps(options['cordon_trigger_parameters'])

        cordon_duration_parameter = options['cordon_duration_parameter']
        group.attrs['cordon_duration_parameter'] = cordon_duration_parameter
        if cordon_duration_parameter == 'stable repair':
            group.attrs['replacement_stabilization_time'] = options['replacement_stabilization_time']

        # record community downtime results
        dset = group.create_dataset('community_downtime', data=community_downtime)
        dset.attrs['downtime_parameters'] = downtime_parameters

        # record community recovery curves
        dset = group.create_dataset('community_recovery', data=community_recovery)
        dset.attrs['time'] = time
        dset.attrs['dt'] = dt
        dset.attrs['recovery_labels'] = ['functional_repair', 'impeding_factor_delay', 'functional_downtime',
                                         'cordon_duration', 'cordon_induced_delay', 'total_delay', 'total_downtime']
        dset.attrs['occupancy_labels'] = occ_labels
        dset.attrs['n_bldgs'] = n_bldgs
        dset.attrs['sqft_totals'] = sqft_totals


        # record cordon logistics
        subgroup_name = 'CordonNetwork'
        subgroup = group.create_group(subgroup_name)
        subgroup_name = group_name + '/' + subgroup_name

        key = subgroup_name + '/cordon_locations'
        cordon_locations.to_hdf(output_filename, key=key, mode='r+')
        dset = subgroup['cordon_locations']
        dset.attrs['cordon_map'] = str(cordon_geojson_generator(cordon_locations))

        key = subgroup_name + '/cordon_adjacency'
        cordon_adjacency.to_hdf(output_filename, key=key, mode='r+')

        dset = subgroup.create_dataset('cordon_durations', data=cordon_durations)
        dset = subgroup.create_dataset('cordon_duration_adjacency', data=cordon_duration_adjacency)

    return community_downtime


def calculate_community_recovery(bldgs, community_downtime, time, downtime_parameters, occ_labels, i_sim):
    # index the columns for each recovery parameter
    recovery_labels = ['functional_repair', 'impeding_factor_delay', 'functional_downtime', 'cordon_duration',
                       'cordon_induced_delay', 'total_delay', 'total_downtime']
    # recovery_idx = [np.where(downtime_parameters == x)[0] for x in recovery_labels]
    recovery_idx = [downtime_parameters.index(x) for x in recovery_labels]

    # retrieve the indices of relevant buildings
    col_occ = 'building.occupancy_id'
    occ_idx = [None] * len(occ_labels)
    occ_idx[0] = bldgs[col_occ] == 'Residential'
    occ_idx[1] = bldgs[col_occ] == 'Commercial Office'
    occ_idx[2] = [True] * len(bldgs)
    occ_n_idx = [np.where(x)[0] for x in occ_idx]

    # retrieve total number and sqft of relevant buildings
    col_sqft = 'building.total_area_ft'
    n_bldgs = [np.sum(x) for x in occ_idx]
    sqft_lists = [bldgs.loc[x, col_sqft].values for x in occ_idx]
    sqft_totals = [np.sum(x) for x in sqft_lists]

    # prepare results matrix
    n_parameters = 2 * len(recovery_idx)
    [n_rups, _, _, _] = community_downtime.shape
    community_recovery = np.ones([len(time), n_rups, len(occ_idx), n_parameters])

    for i_rup in range(n_rups):
        # step through occupancy types
        for i_occ in range(len(occ_idx)):
            # loop through the recovery index for each part of recovery
            for rec_idx, i_rec in zip(recovery_idx, range(len(recovery_idx))):
                # step through time to aggregate the buildings
                t = 0
                n_affected = n_bldgs[i_occ]
                while n_affected > 0:
                    # get the index of the buildings of interest that are still within each recovery index
                    t_idx = np.where(community_downtime[i_rup, occ_n_idx[i_occ], rec_idx, i_sim] > time[t])[0]
                    n_affected = len(t_idx)
                    # 100% - % of buildings
                    parameter_idx = i_rec + len(recovery_idx)
                    community_recovery[t, i_rup, i_occ, parameter_idx] = 1 - len(t_idx) / n_bldgs[i_occ]
                    # 100% - % of sqft
                    parameter_idx = i_rec
                    community_recovery[t, i_rup, i_occ, parameter_idx] = 1 - np.sum(sqft_lists[i_occ][t_idx]) / \
                                                                             sqft_totals[i_occ]
                    t = t + 1

    return community_recovery


def parallel_community_recovery(bldgs, community_downtime, downtime_parameters, occ_labels, dt):
    [_,_,_,n_sims] = community_downtime.shape

    t_min = 0
    t_max = np.max(community_downtime)
    time = np.arange(t_min, t_max+dt, dt)

    # occ_labels = ['Residential', 'Commercial Office', 'All Occupancies']
    p = mp.Pool()
    part_calc_recovery = partial(calculate_community_recovery, bldgs, community_downtime, time, downtime_parameters, occ_labels)
    community_recovery = p.map(part_calc_recovery, [i for i in range(0,n_sims)])
    p.close()
    p.join()

    axis = len(community_recovery[0].shape)
    community_recovery = np.stack([i for i in community_recovery], axis=axis)

    # retrieve the indices of relevant buildings
    col_occ = 'building.occupancy_id'
    occ_labels = ['Residential', 'Commercial Office', 'All Occupancies']
    occ_idx = [None] * len(occ_labels)
    occ_idx[0] = bldgs[col_occ] == 'Residential'
    occ_idx[1] = bldgs[col_occ] == 'Commercial Office'
    occ_idx[2] = [True] * len(bldgs)
    occ_n_idx = [np.where(x)[0] for x in occ_idx]

    # retrieve total number and sqft of relevant buildings
    col_sqft = 'building.total_area_ft'
    n_bldgs = [np.sum(x) for x in occ_idx]
    sqft_lists = [bldgs.loc[x, col_sqft].values for x in occ_idx]
    sqft_totals = [np.sum(x) for x in sqft_lists]

    return community_recovery, time, n_bldgs, sqft_totals
