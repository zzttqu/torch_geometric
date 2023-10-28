from enum import Enum


class StateCode(Enum):
    workcell_ready = 0
    workcell_working = 1
    workcell_low_material = 2
    workcell_low_product = 3
    workcell_function_error = 4
