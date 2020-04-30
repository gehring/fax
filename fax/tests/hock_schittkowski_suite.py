import glob
import os

import fax.test_util

tests_folder = os.path.dirname(__file__)
# tests_folder = os.path.join(, "tests")
models_glob = os.path.join(os.path.dirname(__file__), "*_apm.py")
models = glob.glob(models_glob)
if not models:
    fax.test_util.parse_HockSchittkowski_models(tests_folder)
    models = glob.glob(models_glob)


def load_suite(inequality=False, intermediates=False, ):
    import fax.tests.HockSchittkowski
    for name, cls in fax.tests.HockSchittkowski.__dict__.items():
        if isinstance(cls, type):
            if not name.startswith("Hs") or name == "Hs":
                continue
            if hasattr(cls, "inequality_constraints") and not inequality:
                continue
            if hasattr(cls, "intermediates") and not intermediates:
                continue
            yield cls
