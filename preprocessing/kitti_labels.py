NUMBER_OF_LABELS = 11
DONT_CARE_LABELS = [11]

def getLabel(c):
    r = c[0]
    g = c[1]
    b = c[2]
    if r == 128 and g == 0 and b == 0:
        return 1
    if r == 64 and g == 64 and b == 0:
        return 9
    if r == 128 and g == 128 and b == 0:
        return 5
    if r == 64 and g == 0 and b == 128:
        return 7
    if r == 64 and g == 64 and b == 128:
        return 4
    if r == 128 and g == 64 and b == 128:
        return 2
    if r == 128 and g == 128 and b == 128:
        return 0
    if r == 192 and g == 192 and b == 128:
        return 6
    if r == 0 and g == 0 and b == 192:
        return 3
    if r == 192 and g == 128 and b == 128:
        return 8
    if r == 0 and g == 128 and b == 192:
        return 10
    if r == 0 and g == 0 and b == 0:
        return 11
    return -1
