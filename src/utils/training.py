from time import strftime, gmtime
from torch import Tensor

def get_run_title(model_name: str) -> str:
    """
    Returns a unique run name in format "<specified model name>_<date in YYYY_MM_DD format>__<time in HH_MM format>"
    """
    return f"{model_name.lower().replace(" ", "_")}_{strftime("%Y_%m_%d__%H_%M", gmtime())}"

def one_hot_encode_target(target: int) -> Tensor:
    """
    Converts a number between 0 and 2 to a tensor of length 3 with the target one-hot-encoded.
    """
    output = Tensor([0,0,0])
    output[target] = 1
    return output
