from .base import *


def assign_impeding_factors(community_damage, rc_triggers, if_idx, if_pool, max_rc, weeks):
    # initialize the output
    [n_rups, n_bldgs, _, n_sims] = community_damage.shape
    time = np.zeros([n_rups, n_bldgs, n_sims])

    for i_rc, i_if in zip(rc_triggers, if_idx):
        # find relevant realizations
        idx = np.where(max_rc >= i_rc)

        # identify the relevant realizations within the simulated values
        impeding_factors = if_pool[:, :, i_if, :]

        # assign the relevant realizations
        time[idx] = impeding_factors[idx]

        if weeks:
            days = 7 * time
        else:
            days = time

    return days


def inspection_time(community_damage, if_pool):
    # Inspection is only included if any component reaches the RC=3 repair class
    # This is consistent with REDi v1

    # identify the largest RC for each realization (may be structural or non-structural)
    str_rc_idx = 4
    non_rc_idx = 5
    max_rc = np.maximum(community_damage[:, :, str_rc_idx, :], community_damage[:, :, non_rc_idx, :])

    # prep RC criteria for selecting the relevant impeding factor for FUNCTIONAL recovery
    rc_triggers = [3]
    if_idx = [0]

    # flag for parameters units in weeks
    weeks = 0

    # retrieve the relevant values
    days = assign_impeding_factors(community_damage, rc_triggers, if_idx, if_pool, max_rc, weeks)

    return days


def eng_mob_time(community_damage, if_pool):
    # Engineering Mobilization is determined by the maximum structural repair class OR whether it required a complete redesign
    # This is consistent with REDi v1

    # get max structural RC
    idx = 4
    max_rc = community_damage[:, :, idx, :]

    # set replacement trigger as RC = 4
    idx = 10  ## THIS IDX FOR REPLACEMENT INCLUDES CASES WHERE THE *FULL* REPAIR EXCEEDED REPLACEMENT TIME ##
    replacement = community_damage[:, :, idx, :]
    idx = np.where(replacement == 1)
    max_rc[idx] = 4

    # prep RC criteria for selecting the relevant impeding factor for FUNCTIONAL recovery
    rc_triggers = [3,
                   4]  ## only RC=3 and replacement is relevant for functional recovery, per REDi v1  (structural RCs skip RC=2)
    if_idx = [2, 3]

    # flag for parameters units in weeks
    weeks = 1

    # retrieve the relevant values
    days = assign_impeding_factors(community_damage, rc_triggers, if_idx, if_pool, max_rc, weeks)

    return days


def financing_time(community_damage, if_pool):
    # Financing is based on the maximum repair class of any component
    # This is consistent with REDi v1

    # identify the largest RC for each realization (may be structural or non-structural)
    str_rc_idx = 4
    non_rc_idx = 5
    max_rc = np.maximum(community_damage[:, :, str_rc_idx, :], community_damage[:, :, non_rc_idx, :])

    # prep RC criteria for selecting the relevant impeding factor for FUNCTIONAL recovery
    rc_triggers = [2]
    if_idx = [4]

    # flag for parameters units in weeks
    weeks = 1

    # retrieve the relevant values
    days = assign_impeding_factors(community_damage, rc_triggers, if_idx, if_pool, max_rc, weeks)

    return days


def contr_mob_time(community_damage, if_pool):
    # Contractor Mobilization is based on the maximum repair class of any component
    # This is consistent with REDi v1

    # identify the largest RC for each realization (may be structural or non-structural)
    str_rc_idx = 4
    non_rc_idx = 5
    max_rc = np.maximum(community_damage[:, :, str_rc_idx, :], community_damage[:, :, non_rc_idx, :])

    # prep RC criteria for selecting the relevant impeding factor for FUNCTIONAL recovery
    rc_triggers = [2]
    if_idx = [6]

    # flag for parameters units in weeks
    weeks = 1

    # retrieve the relevant values
    days = assign_impeding_factors(community_damage, rc_triggers, if_idx, if_pool, max_rc, weeks)

    return days


def permitting_time(community_damage, if_pool):
    # Permitting is based on the maximum structural repair class
    # This is consistent with REDi v1

    # get max structural RC
    idx = 4
    max_rc = community_damage[:, :, idx, :]

    # set replacement trigger as RC = 4
    idx = 10  ## THIS IDX FOR REPLACEMENT INCLUDES CASES WHERE THE *FULL* REPAIR EXCEEDED REPLACEMENT TIME ##
    replacement = community_damage[:, :, idx, :]
    idx = np.where(replacement == 1)
    max_rc[idx] = 4

    # prep RC criteria for selecting the relevant impeding factor for FUNCTIONAL recovery
    rc_triggers = [3]
    if_idx = [8]

    # flag for parameters units in weeks
    weeks = 1

    # retrieve the relevant values
    days = assign_impeding_factors(community_damage, rc_triggers, if_idx, if_pool, max_rc, weeks)

    return days