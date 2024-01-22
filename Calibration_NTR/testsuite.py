"""
testsuite.py

Name tests as ut_*.py in base-level directories and subdirectories, and the
test suite will find them.  Assumes the testsuite is in the top level
directory, as it uses its own position to initialize the search.

Optionally takes an additional command-line argument, where it will search
for one of two options:
 - if that name is the name of a unit test file, it will run the tests only in
   that specific file.
 - If that argument gives the name of a valid path relative to the base
   directory, it will run all tests in that directory

"""

import unittest
import os
import sys

if __name__ == "__main__":

    testsuite = unittest.TestSuite()

    localpath = os.path.dirname(os.path.abspath(__file__))
    tl = unittest.defaultTestLoader
    if len(sys.argv) > 1:
        testpath = os.path.join(localpath, sys.argv[1])
        if os.path.isdir(testpath): # specific directory
            testsuite.addTest(tl.discover(testpath, pattern='ut_*.py',
                                          top_level_dir=localpath))
            pass
        else: # specific file
            testsuite.addTest(tl.discover(localpath, pattern=str(sys.argv[1])))
        pass
    else:
        testsuite.addTest(tl.discover(localpath, pattern='ut_*.py'))
        pass
    pass
    unittest.TextTestRunner(verbosity=1, buffer=True).run(testsuite)
