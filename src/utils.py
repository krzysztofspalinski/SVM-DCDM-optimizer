from io import StringIO
import sys
import pandas as pd


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
    if optimizer == 'cvxpy':
        return process_output_cvxpy(output)
    if optimizer == 'quadprog':
        return "unknown", "No result dataframe"


def process_output_cvxopt(output):
    if output[-1].startswith('Terminated'):
        result = False
        output = output[1:-1]
    else:
        result = True
        output = output[1:]

    output = "\n".join([f"{row}".strip() for row in output]).replace(':', " ").replace('  ', ' ').replace(' ', ',')
    df = pd.read_csv(StringIO(output), header=None, sep=',')
    df.columns = ['iteration', 'pcost', 'dcost', 'gap', 'pres', 'dres']
    return result, df


def process_output_cvxpy(output):
    start_idx = [output.index(a) for a in output if a.startswith('iter')][0]
    end_idx = [output.index(a) for a in output if a.startswith('status:')][0] - 1

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
        'iteration': 'int',
        'objective': 'float',
        'pres': 'float',
        'dres': 'float',
        'rho': 'float',
        'time': 'float'
    }

    df = pd.read_csv(StringIO(output), sep=' ', skiprows=1, header=None, names=headers, dtype=dtypes)
    return result, df

