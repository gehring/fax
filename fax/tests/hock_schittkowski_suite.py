import collections
import itertools
import os
import tempfile
import urllib.request
import zipfile
from typing import Text, Union, List

APM_TESTS = "https://apmonitor.com/wiki/uploads/Apps/hs.zip"


def load_suite(inequality=False, intermediates=False, ) -> List["fax.tests.HockSchittkowski"]:
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


def get_list(rows):
    param_list = []
    skipped = []
    for row in rows:
        row_text = row.lstrip()
        if not row_text:
            continue

        if row_text.startswith("!"):
            skipped.append(row_text)
            continue

        if {">=", "<=", "<", ">"}.intersection(row_text):
            raise NotImplementedError("no inequalities")

        if row_text[0].isupper() and row.replace(" ", "").isalpha():
            assert row_text.startswith("End")
            return param_list, skipped
        else:
            param_list.append(row_text)
    raise ValueError


def get_struct(rows):
    struct = {}
    skipped = []
    for row in rows:
        if not row:
            continue

        if row[0] == '!' or row[0] == '#':
            skipped.append(row)
            continue

        row_text = row.lstrip()
        if row_text == "End Model":
            continue

        if row_text[0].isupper():
            struct[row_text], skipped_ = get_struct(rows)
            skipped.extend(skipped_)
        else:
            params, skipped_ = get_list(itertools.chain([row], rows))
            skipped.extend(skipped_)
            return params, skipped
    return struct, skipped


def text_to_code(variable, equation, closure):
    cost_function = equation.replace("^", "**")
    seq = []
    for a in cost_function.split("]"):
        if "[" not in a:
            seq.append(a)
        else:
            rest, num = a.split("[")
            b = f"{rest}[{int(num) - 1}"
            seq.append(b)
    cost_function = "]".join(seq)
    scope = ", ".join(k for k in closure.keys() if not k.startswith("__"))
    cost_function_ = f"{variable} = lambda {scope}: {cost_function}"
    return cost_function_


def apm_to_python(text: Text) -> Union[Text, None]:
    """Convert APM format to a python code file.

    Args:
        text: APM contains of the APM file.
    """

    if "Intermediates" in text:
        raise NotImplementedError("Not implemented yet, maybe never.")
    if "does not exist" in text:
        raise NotImplementedError("I'm not sure how to handle those.")
    rows = iter(text.splitlines())

    struct, skipped = get_struct(rows)

    if len(struct) != 1:
        raise NotImplementedError(f"Found {len(struct)} models in a file, only one is supported.")
    (model_name, model_struct), = struct.items()

    python_code = f"class {model_name.split('Model ')[1].title()}(Hs):\n"

    var_sizes, python_code = _parse_initialization(model_struct, python_code)
    python_code = _parse_equations(model_struct, python_code, var_sizes)

    skipped, python_code = _parse_optimal_solution(python_code, skipped)
    python_code = _parse_constraints(model_struct, python_code)

    python_code = python_code.replace("\t", "    ")
    return python_code


def _parse_constraints(model_struct, python_code):
    constraints = []
    for equation in model_struct['Equations']:
        lhs, rhs = equation.split("=")
        if lhs.strip() != 'obj':
            if not set(rhs.strip()).difference({'0', '.', ','}):
                lhs = f"{lhs} - {rhs}"

            constraint_variable = f"h{len(constraints)}"

            cost_function = text_to_code(constraint_variable, lhs, {'x': None})
            python_code += f"\t{cost_function}\n"
            constraints.append(constraint_variable)

    if constraints:
        python_code += f"""
\tdef constraints(self, x):
\t\treturn stack((self.{'(x), self.'.join(constraints)}(x)))
"""
    return python_code


def _parse_optimal_solution(python_code, skipped):
    for idx, comment in enumerate(skipped):
        if "! best known objective =" in comment:
            _, optimal_solution = comment.split("=")

            python_code += f"\toptimal_solution = -array({optimal_solution.strip()})\n"
            break
    else:
        raise ValueError("No solution found")
    del skipped[idx]
    return skipped, python_code


def _parse_equations(model_struct, python_code, var_sizes):
    for obj in model_struct["Equations"]:
        variable, equation = (o.strip() for o in obj.split("="))

        if "obj" in variable:
            # By default we maximize here.
            equation = "-(" + equation + ")"

            cost_function = text_to_code(variable, equation, var_sizes)
            cost_function = cost_function.replace("obj =", "objective_function =")
            python_code += f"\t{cost_function}\n"
    return python_code


def _parse_initialization(model_struct, python_code):
    var_sizes = collections.defaultdict(int)
    for obj in model_struct["Variables"]:
        if obj == "obj":
            continue

        variable, value = obj.split("=")
        var, size = variable.split("[")
        size, _ = size.split("]")
        if ":" in size:
            size = max(int(s) for s in size.split(":"))
        else:
            size = int(size)

        var_sizes[var] = max(var_sizes[var], size)
    if var_sizes:
        python_code += f"\tinitialize = lambda: (\n"
        for k, v in var_sizes.items():
            python_code += f"\t\tzeros({v}),  # {k}\n"
        python_code += f"\t)\n\n"
    return var_sizes, python_code


def maybe_download_tests(work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, "hs.zip")
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(APM_TESTS, filepath)
        print('Downloaded test file in', work_directory)
    return filepath


def load_HockSchittkowski_models():  # noqa
    for model in load_suite():
        yield model.objective_function, model.constraints, model.optimal_solution, model.initialize


def parse_HockSchittkowski_models(test_folder):  # noqa
    zip_file_path = maybe_download_tests(tempfile.gettempdir())
    if not os.path.exists(test_folder):
        os.makedirs(test_folder, exist_ok=True)

    with open(os.path.join(test_folder, "HockSchittkowski.py"), "w") as test_definitions:

        test_definitions.write("from jax.numpy import *\n")
        test_definitions.write("from jax.config import config\n")
        test_definitions.write('config.update("jax_enable_x64", True)\n\n\n')  # This prevents jax to act up in other code locations
        test_definitions.write("class Hs:\n")
        test_definitions.write("    constraints = lambda: 0\n\n\n")

        with zipfile.ZipFile(zip_file_path) as test_archive:
            for test_case_path in test_archive.filelist:
                try:
                    with test_archive.open(test_case_path) as test_case:
                        python_code = apm_to_python(test_case.read().decode('utf-8'))
                except NotImplementedError:
                    continue
                else:
                    test_definitions.write(python_code + "\n\n\n")


try:
    import fax.tests.HockSchittkowski
except ImportError:
    parse_HockSchittkowski_models(os.path.dirname(__file__))
    import fax.tests.HockSchittkowski
