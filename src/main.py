import os
import sys
import warnings
import numpy as np
import pandas as pd
import mods.utils as utl
import mods.findpeaks as findpeaks
import matplotlib.pyplot as plt



def main(file_name : str):
    warnings.filterwarnings("ignore")
    
    file_without_ext = os.path.basename(file_name)    
    df = pd.read_csv(file_name)
                                                                        
    peak_matrix , fitted , max_at , peak_group_x , peak_group_y ,peak_x_original, peak_y_original = findpeaks.findpeaks(df, 0.4,30 , 20, 10)
    interpolated = utl.data_interpolation(df)

    sample_index , value = utl.find_sample_index(peak_matrix, df)
    print('closest z-pos in sampled data:',sample_index)
    print('closest contrast in sampled data:',value)
    np.set_printoptions(suppress=True)
    print('peak matrix:',peak_matrix)
    
    
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color = 'green')
    plt.scatter(sample_index, value, marker = 'o', color = 'red')
    plt.plot(peak_matrix[:, 1], peak_matrix[:, 2], marker = 'x',linestyle='', color = 'red')
    plt.savefig(f'out/{file_without_ext}.png')
    plt.show()
    
    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Running for all files in ./data")
        # Walk directory and run main() for each file
        for root, dirs, files in os.walk('./data'):
            # Check if file is in .csv format
            for file in files:
                if not file.endswith('.csv'):
                    continue
                print(f'Running for {file}')
                main(os.path.join(root, file))
    else :
        file_name = sys.argv[1]
        main(file_name)    