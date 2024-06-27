import os 

def check_dir(directory):
    """
    Check if a directory exists, and if not, create it.

    Arguments:
    directory -- path of the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)