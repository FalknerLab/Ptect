![](resources/ptect_banner.png)

![](resources/gui_demo.gif)

# Prerequisites
1. A working SLEAP inference/proofreading pipeline for both optical and thermal data
2. Recordings organized as individuals folders which each contain:
    1. Raw optical video ending in _top.mp4
   2. Raw thermal video ending in _thermal.avi
   3. Proofread SLEAP tracking .h5 file for optical data ending in _top.h5 where the number of tracks MUST equal the number of mice
   4. Proofread SLEAP tracking .h5 file for thermal data ending in _thermal.h5 (doesn't care about tracks)
5. Git and Conda for development

# Installation
## Install as a package
`pip install git+https://github.com/FalknerLab/Ptect.git`
### Usage
```
import territorytools as tt

#Running GUI
gui = tt.gui.PtectApp \n

#Processing and Loading Data
data = tt.process.process_all_data(your_folder)
```
## Clone from Source
### Clone repo to local
`git clone https://github.com/FalknerLab/Ptect.git`
### Create TerritoryTools environment and install packages
`conda create --file environment.yml`
### Activate TerritoryTools environment before running code
`conda activate TerritoryTools`

## Running from terminal (requires cloning from source)
After cloning and creating the TT environment, you can process data and run the GUI directly from the terminal.
This is useful for making bash scripts and deploying Ptect to remote machines/clusters



