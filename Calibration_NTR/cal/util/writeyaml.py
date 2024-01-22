"""
Utility function to write YAML files
"""

import yaml

def writeyaml(outdict, path, custom_exception=Exception):
    """
    Write a dictionary to a YAML file located at a given path

    This function will overwrite the contents of the file at that path.  No
    returns.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in the repository.

    Arguments:
     outdict: dictionary to write out to YAML
     path: string containing path to file; can be absolute or relative

    Keyword arguments:
     custom_exception: Exception class to use when raising errors.  Defaults to
      Exception if none is specified.

    """
    if not isinstance(outdict, dict):
        raise TypeError('outdict must be a dict')
    if not isinstance(path, str):
        raise TypeError('path must be a str')

    # Load config from file
    try:
        with open(path, 'w') as FILE:
            yaml.dump(
                outdict,
                FILE,
                default_flow_style=False, # enforces block style
                sort_keys=False, # will dump in dict insertion order on Py3.7+
            )
            pass
        pass
    # Issues with open()
    except OSError:
        raise custom_exception('File could not be opened.')
    # Issues with yaml.dump()
    except yaml.YAMLError: # this is base class for all YAML errors
        raise custom_exception('YAML error raised on write.')

    return
