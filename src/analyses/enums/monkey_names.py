from enum import Enum

class BestFrans(Enum):
    # Best Frans
    B_M1 = "G701"
    B_F1 = "14F"
    B_F2 = "68F"
    B_F3 = "101G"
    B_J1 = "19J"

    RANK = [B_M1, B_F1, B_J1, B_F3, B_F2]


class Zombies(Enum):
    # Zombies
    Z_M1 = "7124"
    Z_F1 = "69X"
    Z_F2 = "72X"
    Z_F3 = "94B"
    Z_F4 = "110E"
    Z_F5 = "67G"
    Z_F6 = "81G"
    Z_F7 = "143H"
    Z_J1 = "87J"
    Z_J2 = "151J"

    RANK = [Z_M1, Z_F3, Z_F1, Z_J1, Z_F2, Z_F7, Z_J2, Z_F4, Z_F5] # excludes 81G


class Instigators(Enum):
    # Instigators
    I_M1 = "G942"
    I_F1 = "35Y"
    I_F2 = "49Y"
    I_F3 = "42Z"
    I_F4 = "48Z"
    I_F5 = "59E"
    I_F6 = "68E"
    I_F7 = "70G"
    I_F8 = "79G"
    I_F9 = "144H"
    I_J1 = "86I"
    I_J2 = "167I"
    I_J3 = "114J"
    I_J4 = "134J"

    RANK = [I_M1, I_F1, I_F2, I_F3, I_F4, I_F5, I_F6, I_F7, I_F8, I_F9, I_J1, I_J2, I_J3, I_J4]


class StrangerThings(Enum):
    # Stranger Things
    S_M1 = "DF2I"
    S_F1 = "68Y"
    S_F2 = "09X"
    S_F3 = "0EX"
    S_F4 = "0FL"
    S_J1 = "37I"
    S_J2 = "58I"
    S_J3 = "46J"
    S_J4 = "26J"
    S_J5 = "36J"
    S_J6 = "40J"

    RANK = [S_M1, S_F1, S_F2, S_F3, S_F4, S_J1, S_J2, S_J3, S_J4, S_J5, S_J6]


def get_monkeys_by_rank(groupname: str):
    group_to_class = {
        "Zombies": Zombies,
        "Best Frans": BestFrans,
        "Instigators": Instigators,
        "Stranger Things": StrangerThings,
    }

    if groupname in group_to_class:
        cls = group_to_class[groupname]  # Get the corresponding class
        return cls.RANK.value  # Get ordered values
    else:
        raise ValueError(f"Unknown group name: {groupname}")
