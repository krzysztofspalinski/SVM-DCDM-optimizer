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


def process_output(output: str) -> (bool, pd.DataFrame):

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
