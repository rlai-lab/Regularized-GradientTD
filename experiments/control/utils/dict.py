def merge(d1, d2):
    ret = d2.copy()
    for key in d1:
        ret[key] = d2.get(key, d1[key])

    return ret
