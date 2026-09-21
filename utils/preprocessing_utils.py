def get_night(
        data,
        target_obsid,
    ):
    """
    Given a dictionary mapping obsids to pointings, and the starting obsid,
    return all adjacent obsids for that night and pointing.

    Parameters:
        data (dict): Dictionary of obsid strings mapped to integer values.
        target_obsid (str): The obsid to search around.

    Returns:
        list: A list of adjacent obsids (including the target) from the same night
    """
    sorted_obsids = sorted(data.keys(), key=int)
    target_index = sorted_obsids.index(target_obsid)
    target_value = data[target_obsid]
    result = [target_obsid]

    # Go left
    i = target_index - 1
    while i >= 0 and data[sorted_obsids[i]] == target_value:
        result.insert(0, sorted_obsids[i])
        i -= 1

    # Go right
    i = target_index + 1
    while i < len(sorted_obsids) and data[sorted_obsids[i]] == target_value:
        result.append(sorted_obsids[i])
        i += 1

    return target_value, result  # return the pointing and all associated obsids for that night


def get_pointing(
        data,
        pointing,
    ):
    """
    Given a dictionary mapping obsids to pointings, and the desired pointing,
    return *all* obsids for that pointing.

    Parameters:
        data (dict): Dictionary of obsid strings mapped to integer values.
        pointing (int): The desired pointing

    Returns:
        list: Complete list of obsids from that poiting
    """
    result = []
    for i in data:
        if data[i] == pointing:
            result.append(i)

    return result  # return all associated obsids for that pointing


def get_ref_obsids(
        data,
    ):
    """
    Given a data dictionary that maps OBSIDs to pointings,
    get the list of all 'reference' OBSIDs; i.e. the first
    OBSID per pointing.
    """
    # Sort the OBSIDs numerically (they should already be sorted, but just in case)
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))

    ref_obsids = []
    prev_pointing = None

    for obsid, pointing in sorted_items:
        if pointing != prev_pointing:
            ref_obsids.append(obsid)
            prev_pointing = pointing

    return ref_obsids
