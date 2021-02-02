import sys
from io import StringIO

import pandas as pd

from dcdm import DCDM


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def process_output(output: str, optimizer) -> (bool, pd.DataFrame):

    if optimizer == 'cvxopt':
        return process_output_cvxopt(output)
    elif optimizer == 'osqp':
        return process_output_osqp(output)
    elif optimizer == 'cvxpy_OSQP':
        return process_output_cvxpy_osqp(output)
    elif optimizer == 'cvxpy_SCS':
        return process_output_cvxpy_scs(output)
    elif optimizer == 'cvxpy_GUROBI':
        return process_output_cvxpy_gurobi(output)
    elif optimizer == 'cvxpy_MOSEK':
        return process_output_cvxpy_mosek(output)
    elif optimizer == 'ecos':
        return process_output_ecos(output)
    elif optimizer == 'gurobi':
        return process_output_gurobi(output)
    elif optimizer == 'mosek':
        return process_output_mosek(output)
    elif optimizer == 'quadprog':
        return "unknown", "No result dataframe", None
    elif optimizer == DCDM:
        return process_output_dcdm(output)
    else:
        return "unknown", "\n".join(output), None


def process_output_cvxpy_osqp(output):
    runtime = float(output[-3].split()[-1][:-1])
    if output[-1].startswith('Terminated'):
        result = False
        output = output[1:-1]
    else:
        result = True
        output = output[1:-1]
    output = "\n".join(output[15:-7])
    df = pd.read_csv(StringIO(output), header=0,
                     sep='\s{2,}', engine='python', na_values=['--------'])
    return result, df, runtime


def process_output_osqp(output):
    runtime = float(output[-3].split()[-1][:-1])
    if output[-1].startswith('Terminated'):
        result = False
        output = output[1:-1]
    else:
        result = True
        output = output[1:-1]
    output = "\n".join(output[15:-7])
    df = pd.read_csv(StringIO(output), header=0,
                     sep='\s{2,}', engine='python', na_values=['--------'])
    return result, df, runtime


def process_output_cvxopt(output):
    if output[-1].startswith('Terminated'):
        result = False
        output = output[1:-1]
    else:
        result = True
        output = output[1:-1]
    output = "\n".join([l.replace(':', '') for l in output])
    df = pd.read_csv(StringIO(output), header=None, sep='\s+')
    df.columns = ['iteration', 'pcost', 'dcost', 'gap', 'pres', 'dres']
    return result, df, None


def process_output_cvxpy_scs(output):
    runtime = float(output[-13].split()[-1][:-1])
    if output[-14].endswith('Solved'):
        result = True
    else:
        result = False
    columns = output[13].split(' | ')
    output = output[15:-15]
    output = "\n".join(output)
    # print(output)
    # raise Exception
    df = pd.read_csv(StringIO(output), header=None,
                     sep='\s*\|?\s+', na_values=['na'], engine="python")

    df.columns = columns
    return result, df, runtime


def process_output_dcdm(output):
    result = True
    output = "\n".join(output)
    df = pd.read_csv(StringIO(output), header=0, sep='\s+', na_values=['na'])
    df.columns = ['iteration', 'pcost', 'dcost', 'pres', 'dres']
    return result, df, None


def process_output_ecos(output):
    runtime = float(output[-2].split()[-2])
    if output[-3].startswith('OPTIMAL'):
        result = True
    else:
        result = False
    output = "\n".join([l[:-17] for l in output[3:-3]])
    df = pd.read_csv(StringIO(output), header=0, sep='\s+', na_values=['---'])
    return result, df, runtime


def process_output_gurobi(output):
    if "solved" in output[-3]:
        result = True
    else:
        result = False
    output = "\n".join(output[28:-4])
    df = pd.read_csv(StringIO(output), header=None,
                     sep='\s+', na_values=['na'])
    df.columns = ['Iter', 'Obj_prim', 'Obj_dual',
                  'Res_prim', 'Res_Dual', 'Compl', 'Time']
    return result, df


def process_output_cvxpy_gurobi(output):
    runtime = float(output[-3].split()[-2])
    if "solved" in output[-3]:
        result = True
    else:
        result = False
    output = "\n".join(output[30:-4])
    df = pd.read_csv(StringIO(output), header=None,
                     sep='\s+', na_values=['na'])
    df.columns = ['Iter', 'Obj_prim', 'Obj_dual',
                  'Res_prim', 'Res_Dual', 'Compl', 'Time']
    return result, df, runtime


def process_output_mosek(output):
    runtime = float(output[-8].split()[-1])
    if output[-3].endswith('OPTIMAL'):
        result = True
    else:
        result = False
    output = "\n".join(output[43:-8])
    df = pd.read_csv(StringIO(output), header=0,
                     sep='\s+', na_values=['na'])
    return result, df, runtime


def process_output_cvxpy_mosek(output):
    runtime = float(output[-8].split()[-1])
    if output[-3].endswith('OPTIMAL'):
        result = True
    else:
        result = False
    output = "\n".join(output[33:-8])
    df = pd.read_csv(StringIO(output), header=0,
                     sep='\s+', na_values=['na'])
    return result, df, runtime


def process_output_cvxpy(output):
    start_idx = [output.index(a) for a in output if a.startswith('iter')][0]
    end_idx = [output.index(a)
               for a in output if a.startswith('status:')][0] - 1

    if output[end_idx + 1].endswith('solved'):
        result = True
    else:
        result = False
    output = output[start_idx:end_idx]

    for i, text in enumerate(output):
        if text.endswith('s'):
            output[i] = text.replace('s', '')

    output = "\n".join([f"{row}".strip() for row in output])
    while '  ' in output:
        output = output.replace('  ', ' ')  # remove multiple spaces

    headers = ['iteration', 'objective', 'pres', 'dres', 'rho', 'time [s]']
    dtypes = {
        'objective': 'float',
        'pres': 'float',
        'dres': 'float',
        'rho': 'float',
        'time': 'float'
    }

    df = pd.read_csv(StringIO(output), sep=' ', skiprows=1,
                     header=None, names=headers, dtype=dtypes, na_values=['--------'])
    return result, df
