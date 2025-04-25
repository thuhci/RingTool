from enum import Enum


class ExperimentMode(Enum):
    """Enumeration of available experiment modes."""
    TRAIN = "train"
    TEST = "test"
    FIVE_FOLD = "5fold"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, string: str):
        if string in cls._value2member_map_:
            return cls(string)
        else:
            raise ValueError(f"Invalid experiment mode: {string}. Choose from {list(cls._value2member_map_.keys())}.")
