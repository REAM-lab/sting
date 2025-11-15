import pandas as pd
from scipy.linalg import eigvals
import numpy as np

def modal_analisis(A : np.ndarray, 
                   show : bool = False, 
                   print_settings : dict = {'index' : True, 
                                            'tablefmt': "grid",
                                            'numalign': "right", 
                                            'floatfmt': '.3f'}):
    '''Computes eigenvalues, natural frequency, damping ratio, time constant. It also has the option to display a
    pretty table when the function is executed.
    
    Args:
    ----
    A (numpy array): Matrix A of state-space model:
    
    show (Boolean): True (print table), False (do not print). By default is False.
    
    print_settings (dict): setting applied to tabulate package to print the pandas dataframe.

    Returns:
    -------

    df (Dataframe) : It contains eigenvalues, real, imag parts, natural frequency, damping ratio, and time constant.
    
    '''

    eigenvalues = eigvals(A)

    df = pd.DataFrame(data=eigenvalues, columns= ['eigenvalue'])
    df['real'] = df.apply(lambda row: row['eigenvalue'].real, axis=1)
    df['imag'] = df.apply(lambda row: row['eigenvalue'].imag, axis=1)
    df['natural_frequency'] = df.apply(lambda row: abs(row['eigenvalue']/(2*np.pi)), axis=1)
    df['damping_ratio'] =  df.apply(lambda row: -row['eigenvalue'].real/(abs(row['eigenvalue'])), axis=1)
    df['time_constant'] = df.apply(lambda row: -1/row['eigenvalue'].real, axis=1)
    df = df.sort_values(by='real', ascending=False, ignore_index=True)


    if show:
        df_to_print = df.copy()
        df_to_print = df_to_print[['real', 'imag', 'damping_ratio', 'natural_frequency', 'time_constant']]
        df_to_print.rename(columns={'real': 'Eigenvalue \n real part',
                                    'imag': 'Eigenvalue \n imaginary part',
                                    'damping_ratio': 'Damping \n ratio [p.u.]', 
                                    'natural_frequency': 'Natural \n frequency [Hz]',
                                    'time_constant': 'Time \n constant [s]'}, inplace=True)
        print(df_to_print.to_markdown(**print_settings))

    return df

