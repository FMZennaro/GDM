
import tabulate as tab

def print_latex_usertable(D):
    print(tab.tabulate(D, tablefmt="latex", floatfmt=".2f"))