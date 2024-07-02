from slsne.mosutils import process_mosfit
from slsne.utils import get_use_names, get_data_table
import os

# Import FLEET data
use_names = get_use_names()
data_table = get_data_table()

# Process the MOSFiT data of each object
for i in range(len(use_names)):
    print(i + 1, '/', len(use_names))
    object_name = use_names[i]

    if not os.path.exists(f'{object_name}/jupyter/{object_name}_bol.txt'):
        # Match object data
        match = data_table['Name'] == object_name
        redshift = data_table['Redshift'][match][0]

        output_dir = f'{object_name}/jupyter'

        process_mosfit(object_name, mosfit_dir='../mosfit_data', output_dir=output_dir,
                       data_table=data_table, redshift = redshift, plot_parameters=True, plot_corner=True,
                       plot_lc=True, plot_bol=True, calc_rest=True, save_rest_frame=True)
