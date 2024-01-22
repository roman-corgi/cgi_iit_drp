import os
import unittest
from cal.util.abs_rel_path import abs_or_rel_path
from cal.util.loadyaml import loadyaml

LOCAL_PATH = os.path.abspath(os.path.dirname( __file__ ))


class TestAbsOrRelPath(unittest.TestCase):
    """
    Unit tests for abs_or_rel_path function.
    """
    def setUp(self):
        self.input_file = os.path.join(LOCAL_PATH,
                                    'excam_config.yaml')

    def test_load_abs_rel(self):
        """Verify we can load from both absolute and relative paths"""

        excam_dict = loadyaml(self.input_file)
        rel_path = 'excam_config.yaml'
        excam_dict_rel = loadyaml(abs_or_rel_path(rel_path))
        # testing the specification of optional rel input
        excam_dict_rel2 = loadyaml(abs_or_rel_path(rel_path,
                                                            rel=LOCAL_PATH))
        self.assertDictEqual(excam_dict, excam_dict_rel)
        self.assertDictEqual(excam_dict, excam_dict_rel2)

    if __name__ == '__main__':
        unittest.main()
