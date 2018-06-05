## Installation Guide  
  
### Prerequisites  
  
Python 3 with `pip3` is needed.  
  
### Installation  
  
1. Download the source code from this repo.  
2. Run the following bash script to install this repo. Dependencies will be installed automatically. 

       pip3 install /path/of/the/downloaded/folder

## Usage

Run `"run.py -h"` from the source folder to get a short help message. Then use `"run.py <command> -h"` for detailed information of specified `<command>`.

An example data file is provided in the source folder, which is named `"37C_CTL_BCL-500.dat"`. It is actually a .csv file with '\t' as the delimiter. If you would like to use data processing functions such as the `"measure"` or `"clean"` command, you have to make the output of your simulation have the same format of the example file.