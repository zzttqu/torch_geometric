from enum import Enum


class CellCode(Enum):
    workcell_ready = 0
    workcell_working = 1
    workcell_low_material = 2
    workcell_function_error = 3


class CenterCode(Enum):
    center_ready = 0
    center_working = 1
