import os
from cal.util import check

def abs_or_rel_path(path, rel=None):
    """
    Takes what might be an absolute or relative path and returns an
    absolute one.  What the path is relative to is specified by the
    optional input rel, which must be an absolute path itself.
    """
    if rel is None:
        rel = os.path.dirname(os.path.realpath(__file__))
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(rel, path)