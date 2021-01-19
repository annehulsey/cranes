from .base import *
from .community_damage_sampling import *
from .downtime_logistics import *

def load_results(input_filenames, output_filename, i_analysis, options):
    ####

    [i_damage, i_impeding_factors, i_cordons] = i_analysis

    if input_filenames is None:
        new_analysis = False
    else:
        new_analysis = True
        [inventory_filename, ground_motion_filename, original_vulnerability_filename, retrofit_vulnerability_filename] = input_filenames


    # if output folder does not yet exist, create it
    if not os.path.exists(output_filename):
        if new_analysis:
            _ = h5py.File(output_filename, 'w')
            # prepare a group folder for results
            with h5py.File(output_filename, 'r+') as hf:
                _ = hf.create_group('Results')
            print('File created')
        else:
            print('Analysis file does not exist, specify inputs')
            return

    # if i_damage results do not yet exist, sample and save the damage
    damage_name = 'CommunityDamage_' + str(i_damage)
    with h5py.File(output_filename, 'r+') as hf:
        if damage_name in hf['Results'].keys():
            community_damage = hf['Results'][damage_name]['community_damage'][:]
            print('Damage loaded')

        elif new_analysis:
            if options['n_realizations'] is None:
                print('specify the number of realizations')
                return
            community_damage = sample_community_damage_with_retrofits(inventory_filename, ground_motion_filename,
                                                       original_vulnerability_filename, retrofit_vulnerability_filename,
                                                       output_filename, i_analysis, options)
            print('Damage sampled')

        else:
            print('Damage scenario does not exist in analysis file')
            return


    # if i_impeding_factors do not yet exist, sample and save impeding factors
    if_name = 'ImpedingFactors_' + str(i_impeding_factors)
    with h5py.File(output_filename, 'r+') as hf:
        if if_name in hf['Results'][damage_name]['DowntimeLogistics'].keys():
            impeding_factors = hf['Results'][damage_name]['DowntimeLogistics'][if_name]['impeding_factors'][:]
            print('Impeding factors loaded')
        elif new_analysis:
            bldgs = pd.read_hdf(output_filename, key='MetaData/buildings', mode='r+')
            impeding_factors = sample_impeding_factors(bldgs, community_damage, output_filename, i_analysis, options)
            print('Impeding factors sampled')
        else:
            print('Impeding factors scenario does not exist in analysis file')
            return

    # if i_cordons do not yet exist, sample and save cordons
    cordon_name = 'Cordons_' + str(i_cordons)
    with h5py.File(output_filename, 'r+') as hf:
        if cordon_name in hf['Results'][damage_name]['DowntimeLogistics'][if_name]['CordonLogistics'].keys():
            bldgs = pd.read_hdf(output_filename, key='MetaData/buildings', mode='r+')
            community_downtime = \
            hf['Results'][damage_name]['DowntimeLogistics'][if_name]['CordonLogistics'][cordon_name][
                    'community_downtime'][:]
            print('Cordons loaded')
        elif new_analysis:
            bldgs = pd.read_hdf(output_filename, key='MetaData/buildings', mode='r+')
            community_downtime = evaluate_cordons(bldgs, community_damage, impeding_factors,
                                                  output_filename, i_analysis, options)
            print('Cordons evaluated')
        else:
            print('Cordon scenario does not exist in analysis file')
            return

    print()
    return bldgs, community_damage, community_downtime


def plot_percentile_community_recovery(community_recovery, time, xlim, i_rup, i_occ):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    q = [5, 33, 50, 67, 95]
    q_idx = [[0, 4], [1, 3], [2]]
    q_labels = [str(q[0]) + ' and ' + str(q[4]) + 'th %', \
                str(q[1]) + ' and ' + str(q[3]) + 'th %', \
                'Median']
    linestyles = [':', '--', '-']
    linewidths = [1, 1.5, 3]

    occ_labels = ['Residential', 'Commercial Office', 'All Occupancies']
    recovery_labels = ['functional_repair', 'impeding_factor_delay', 'functional_downtime', 'cordon_duration',
                       'cordon_induced_delay', 'total_delay', 'total_downtime']
    repair_idx = recovery_labels.index('functional_repair')
    no_cordon_downtime_idx = recovery_labels.index('functional_downtime')
    downtime_idx = recovery_labels.index('total_downtime')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[2], colors[0], colors[1]]

    label_idx = [repair_idx, no_cordon_downtime_idx, downtime_idx]
    label_names = ['Repair Time Only', 'Impeding Factor Delays', 'Downtime due to Cordons']
    for i in range(len(label_idx)):
        color = colors[i]
        rec_idx = label_idx[i]
        label_name = label_names[i]

        recovery_curves = np.transpose(
            100 * np.percentile(community_recovery[:, i_rup, i_occ, rec_idx, :], q=q, axis=1))

        for j in range(len(q_idx)):
            idx = q_idx[j]
            _ = plt.plot(time, recovery_curves[:, idx], color=color, label=label_name, linestyle=linestyles[j],
                         linewidth=linewidths[j])

    legend_elements = [
        Line2D([0], [0], label=label_names[0], color=colors[0], linewidth=linewidths[-1], linestyle=linestyles[-1]),
        Line2D([0], [0], label=label_names[1], color=colors[1], linewidth=linewidths[-1], linestyle=linestyles[-1]),
        Line2D([0], [0], label=label_names[2], color=colors[2], linewidth=linewidths[-1], linestyle=linestyles[-1]),
        Line2D([0], [0], color='none'),
        Line2D([0], [0], label=q_labels[-1], color='gray', linewidth=linewidths[-1], linestyle=linestyles[-1]),
        Line2D([0], [0], label=q_labels[1], color='gray', linewidth=linewidths[1], linestyle=linestyles[1]),
        Line2D([0], [0], label=q_labels[0], color='gray', linewidth=linewidths[0], linestyle=linestyles[0])]
    _ = plt.legend(handles=legend_elements)

    _ = plt.xlabel('Days after the earthquake')
    _ = plt.ylabel(occ_labels[i_occ] + ',' + '\n' + '% of pre-event sqft', multialignment='center')
    _ = plt.ylim([0, 100])
    _ = plt.xlim(xlim)

    _ = plt.grid(axis='both', linestyle='--', color='lightgray')
    _ = plt.show()


def plot_mean_community_recovery(community_recovery, time, xlim, i_rup, i_occ, sqft_totals, title):
    fig, ax = plt.subplots(figsize=((10, 5)))

    occ_labels = ['Residential', 'Commercial Office', 'All Occupancies']
    recovery_labels = ['functional_repair', 'impeding_factor_delay', 'functional_downtime', 'cordon_duration',
                       'cordon_induced_delay', 'total_delay', 'total_downtime']
    repair_idx = recovery_labels.index('functional_repair')
    no_cordon_downtime_idx = recovery_labels.index('functional_downtime')
    downtime_idx = recovery_labels.index('total_downtime')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    idx = repair_idx
    repair_time = 100 * np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    plt.fill_between(time, repair_time, 100, color='darkgray')

    idx = no_cordon_downtime_idx
    no_cordon_downtime = 100 * np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    if not np.array_equal(no_cordon_downtime, repair_time):
        plt.fill_between(time, no_cordon_downtime, repair_time, color=colors[0])

    idx = downtime_idx
    downtime = 100 * np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    if not np.array_equal(downtime, no_cordon_downtime):
        plt.fill_between(time, downtime, no_cordon_downtime, color=colors[1])

    add_downtime_contributions_bar(community_recovery, time, i_rup, i_occ, sqft_totals, ax, xlim)

    legend_elements = [Patch(facecolor=colors[0], label='Impeding Factors'),
                       Patch(facecolor=colors[1], label='Cordon Delays'),
                       Patch(facecolor='darkgray', label='Repair Time')
                       ]
    _ = plt.legend(handles=legend_elements, title='   Average Contribution of:  ', loc=(0.59, 0.02))

    _ = ax.set_xlabel('Days after the earthquake')
    #     _ = ax.set_ylabel(occ_labels[i_occ] + ',' + '\n' + '% of pre-event sqft' + '\n' + '(out of ' + '{:.1e}'.format(sqft_totals[i_occ]) + 'sqft)', multialignment='center')
    _ = ax.set_ylabel(occ_labels[i_occ] + ',' + '\n' + '% of pre-event sqft', multialignment='center')

    _ = ax.set_ylim([0, 100])
    _ = ax.set_xlim(xlim)
    _ = plt.title(title)

    _ = plt.grid(axis='both', linestyle='--', color='lightgray')
    _ = plt.show()


def add_downtime_contributions_bar(community_recovery, time, i_rup, i_occ, sqft_totals, ax, xlim):
    time_frame = 360
    #     metric = 'sqft-days'
    metric = 'community days'

    recovery_labels = ['functional_repair', 'impeding_factor_delay', 'functional_downtime', 'cordon_duration',
                       'cordon_induced_delay', 'total_delay', 'total_downtime']
    repair_idx = recovery_labels.index('functional_repair')
    no_cordon_downtime_idx = recovery_labels.index('functional_downtime')
    downtime_idx = recovery_labels.index('total_downtime')

    [n_time, n_rups, n_blgs, n_parameters, n_sims] = community_recovery.shape

    if time_frame is not None:
        n_time = np.where(time == time_frame)[0][0] + 1
        time = time[:n_time]
        community_recovery = community_recovery[:n_time, :, :, :]

    baseline_time = np.trapz(np.ones(n_time), time)

    idx = repair_idx
    repair_time = np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    repair_time = baseline_time - np.trapz(repair_time, time)

    idx = no_cordon_downtime_idx
    impeding_factor_time = np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    impeding_factor_time = baseline_time - np.trapz(impeding_factor_time, time) - repair_time

    idx = downtime_idx
    cordon_time = np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    cordon_time = baseline_time - np.trapz(cordon_time, time) - repair_time - impeding_factor_time

    idx = downtime_idx
    downtime = np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    downtime = (baseline_time - np.trapz(downtime, time))

    delays = [impeding_factor_time, cordon_time, repair_time]
    proportional_delay = []
    for idx in range(len(delays)):
        proportional_delay.append(delays[idx] / downtime)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ylim = [0, 100]
    y_start = 0.382 * ylim[1]
    y_height = 0.1 * ylim[1]
    y_mid = (2 * y_start + y_height) / 2
    x_start = 0.59 * xlim[1]
    x_width = 0.99 * xlim[1] - x_start
    patches = []
    for idx in range(len(delays)):
        width = proportional_delay[idx] * x_width
        rect = Rectangle((x_start, y_start), width, y_height)
        patches.append(rect)
        x_mid = (2 * x_start + width) / 2
        x_start = x_start + width
        if width > 0:
            if False:
                ax.text(x_mid, y_mid, ('{:.0f}'.format(100 * proportional_delay[idx]) + '%'), ha='center', va='center',
                        color='white', fontsize='medium', fontweight='bold', zorder=20)
            else:
                ax.text(x_mid, y_mid, ('{:.0f}'.format(delays[idx])), ha='center', va='center',
                        color='white', fontsize='medium', fontweight='bold', zorder=20)

    # total downtime patch
    x_start = 0.59 * xlim[1]
    x_width = 0.99 * xlim[1] - x_start
    width = x_width
    y_start = y_start + y_height + 1.5
    y_mid = (2 * y_start + y_height) / 2
    rect = Rectangle((x_start, y_start), width, y_height)
    patches.append(rect)
    x_mid = (2 * x_start + width) / 2
    if metric == 'sqft-days':
        downtime = downtime * sqft_totals[i_occ]
        ax.text(x_mid, y_mid, ('{:.2e}'.format(downtime) + ' sqft-days'), ha='center', va='center', color='white',
                fontsize='medium', fontweight='bold', zorder=20)
    else:
        ax.text(x_mid, y_mid, ('{:.0f}'.format(downtime) + ' community days'), ha='center', va='center', color='white',
                fontsize='medium', fontweight='bold', zorder=20)

    pc = PatchCollection(patches, facecolor=[colors[0], colors[1], 'darkgray', 'tab:gray'], zorder=10)
    _ = ax.add_collection(pc)

    # downtime deaggregation label
    y_start = y_start + y_height
    y_mid = (2 * y_start + y_height) / 2
    x_mid = (2 * x_start + width) / 2
    if time_frame is None:
        ax.text(x_mid, y_mid, ('Total loss:'), ha='center', va='center', color='black', fontsize='large', zorder=20)
    else:
        ax.text(x_mid, y_mid, ('Loss in first ' + '{:.0f}'.format(time[-1]) + ' days:'), ha='center', va='center',
                color='black', fontsize='large', zorder=20)


def grid_plot_mean_community_recovery(community_recovery, time, xlim, i_rup, i_occ, sqft_totals, ax):
    occ_labels = ['Residential', 'Commercial Office', 'All Occupancies']
    recovery_labels = ['functional_repair', 'impeding_factor_delay', 'functional_downtime', 'cordon_duration',
                       'cordon_induced_delay', 'total_delay', 'total_downtime']
    repair_idx = recovery_labels.index('functional_repair')
    no_cordon_downtime_idx = recovery_labels.index('functional_downtime')
    downtime_idx = recovery_labels.index('total_downtime')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    idx = repair_idx
    repair_time = 100 * np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    ax.fill_between(time, repair_time, 100, color='darkgray')

    idx = no_cordon_downtime_idx
    no_cordon_downtime = 100 * np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    if not np.array_equal(no_cordon_downtime, repair_time):
        ax.fill_between(time, no_cordon_downtime, repair_time, color=colors[0])

    idx = downtime_idx
    downtime = 100 * np.mean(community_recovery[:, i_rup, i_occ, idx, :], axis=1)
    if not np.array_equal(downtime, no_cordon_downtime):
        ax.fill_between(time, downtime, no_cordon_downtime, color=colors[1])

    add_downtime_contributions_bar(community_recovery, time, i_rup, i_occ, sqft_totals, ax, xlim)

    legend_elements = [Patch(facecolor=colors[0], label='Impeding Factors'),
                       Patch(facecolor=colors[1], label='Cordon Delays'),
                       Patch(facecolor='darkgray', label='Repair Time')
                       ]
    _ = ax.legend(handles=legend_elements, title='   Average Contribution of:  ', loc=(0.59, 0.02))

    _ = ax.grid(axis='both', color='lightgray', alpha=0.5)