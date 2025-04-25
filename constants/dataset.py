from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, string: str):
        if string in cls._value2member_map_:
            return cls(string)
        else:
            raise ValueError(f"Invalid dataset type: {string}. Choose from {list(cls._value2member_map_.keys())}.")


# Constants for dataset configuration
ACCEL_CHANNELS = {
    "ax-raw", "ax-filtered", "ax-standardized", "ax-difference", "ax-welch", "ax-filtered-rr", "ax-welch-rr",
    "ay-raw", "ay-filtered", "ay-standardized", "ay-difference", "ay-welch", "ay-filtered-rr", "ay-welch-rr",
    "az-raw", "az-filtered", "az-standardized", "az-difference", "az-welch", "az-filtered-rr", "az-welch-rr"
}

AVAILABLE_TASKS = [
    'hr', 'bvp_hr', 'bvp_sdnn', 'bvp_rmssd',
    'bvp_nn50', 'bvp_pnn50', 'resp_rr', 'spo2', 'samsung_hr', 'oura_hr',
    'BP_sys', 'BP_dia'
]

ALL_SCENARIOS = [
    "sitting", "spo2", "deepsquat", "talking", 
    "shaking_head", "standing", "striding"
]
