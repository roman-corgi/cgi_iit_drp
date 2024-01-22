"""Remove all data files generated when running the TDD GSW VIs.""" 
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cleanup():
    
    subfolder_list = [
        'input_data/',
        'temp_data/',
        'output/',
        'temp_data/',       
    ]

    for subfolder in subfolder_list:
        # Remove all files from the subfolder except readme.txt
        _ =  subprocess.run(
            ("find %s -maxdepth 1 -type f \! -name 'readme.txt' -delete" % (subfolder)),
            stdout=subprocess.PIPE, shell=True)


if __name__ == '__main__':
    run_cleanup()
