def to_256(tensor):
    return (tensor + 1) * 127.5


def to_1(tensor):
    return (tensor / 127.5) - 1
