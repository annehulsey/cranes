from .base import *

def inspection_parameters(mitigation, if_dictionary):
    medians = np.empty(len(mitigation))
    betas = np.empty(len(mitigation))

    for i in list(if_dictionary['Inspection'].keys()):
        for j in list(if_dictionary['Inspection'][i].keys()):
            idx = np.where((mitigation.facility == i) & (mitigation.inspection == j))
            medians[idx] = if_dictionary['Inspection'][i][j]['median']
            betas[idx] = if_dictionary['Inspection'][i][j]['beta']

    return medians, betas


def eng_mob_parameters(mitigation, if_dictionary):
    medians = np.empty([len(mitigation), 3])
    betas = np.empty([len(mitigation), 3])

    for i in list(if_dictionary['EngMob'].keys()):
        idx = np.where(mitigation.eng_mob == i)
        medians[idx, 0] = if_dictionary['EngMob'][i]['Low Damage']['median']
        betas[idx, 0] = if_dictionary['EngMob'][i]['Low Damage']['beta']
        medians[idx, 1] = if_dictionary['EngMob'][i]['High Damage']['median']
        betas[idx, 1] = if_dictionary['EngMob'][i]['High Damage']['beta']
        medians[idx, 2] = if_dictionary['EngMob'][i]['Replacement']['median']
        betas[idx, 2] = if_dictionary['EngMob'][i]['Replacement']['beta']

    return medians, betas


def financing_parameters(mitigation, if_dictionary):
    medians = np.empty([len(mitigation)])
    betas = np.empty([len(mitigation)])

    for i in list(if_dictionary['Financing'].keys()):
        idx = np.where(mitigation.financing == i)
        medians[idx] = if_dictionary['Financing'][i]['median']
        betas[idx] = if_dictionary['Financing'][i]['beta']

    return medians, betas


def contr_mob_parameters(mitigation, if_dictionary):
    medians = np.empty([len(mitigation), 2])
    betas = np.empty([len(mitigation), 2])

    # amend the facility marker to include "Highrise"
    Facility = mitigation.facility.values
    Facility[np.where(mitigation['building.num_stories'] >= 20)] = 'Highrise'

    for i in list(if_dictionary['ContrMob'].keys()):
        for j in list(if_dictionary['ContrMob'][i].keys()):
            idx = np.where((Facility == i) & (mitigation.contr_mob == j))
            medians[idx, 0] = if_dictionary['ContrMob'][i][j]['Low Damage']['median']
            betas[idx, 0] = if_dictionary['ContrMob'][i][j]['Low Damage']['beta']
            medians[idx, 1] = if_dictionary['ContrMob'][i][j]['High Damage']['median']
            betas[idx, 1] = if_dictionary['ContrMob'][i][j]['High Damage']['beta']

    return medians, betas


def permitting_parameters(mitigation, if_dictionary):
    medians = np.empty([len(mitigation), 2])
    betas = np.empty([len(mitigation), 2])

    for i in list(if_dictionary['Permitting'].keys()):
        idx = np.where(mitigation.permitting == i)
        medians[idx, 0] = if_dictionary['Permitting'][i]['Low Damage']['median']
        betas[idx, 0] = if_dictionary['Permitting'][i]['Low Damage']['beta']
        medians[idx, 1] = if_dictionary['Permitting'][i]['High Damage']['median']
        betas[idx, 1] = if_dictionary['Permitting'][i]['High Damage']['beta']

    return medians, betas