from time import strftime, gmtime

def get_run_title(model_name: str) -> str:
    """
    Returns a unique run name in format "<specified model name>_<date in YYYY_MM_DD format>__<time in HH_MM format>"
    """
    return f"{model_name.lower().replace(" ", "_")}_{strftime("%Y_%m_%d__%H_%M", gmtime())}"
