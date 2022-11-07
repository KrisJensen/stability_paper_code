## Long-term stability of neural activity in the motor system

Code for reproducing analyses in Jensen et al. (2022).

### Reproducing results

To download partially processed data and reproduce all analyses and figures:

`conda create -n stability --file requirements.txt`\
`conda activate stability`\
`pip install tensorflow==2.7`\
`pip install gdown`\
`bash download_data.sh`\
`python run_everything.py`

### Data structure

Note that all data have previously been used by Dhawale et al. (2017, 2021); see those papers for access to the full data. After following the installation and download steps above, the data used in Jensen et al. (2022) can be accessed in python as follows:\
`parse_rat.py ratname`\
`from utils import load_rat`\
`rat = load_rat('ratname')`

The dataset consists of 6 rats: `Hindol`, `Dhanashri`, and `Jaunpuri` (DLS), and `Hamir`, `Gorakh` and `Gandhar` (MC).

`load_rat('ratname')` returns a dictionary with the following contents:

_rat['trials']_: behavioral data for each day of recording (keys)

> _rat['trials'][rec_day]['ipis']_: IPI for each trial (in seconds).

> _rat['trials'][rec_day]['modes']_: Behavioral cluster that the trial belongs to (c.f. Methods).

> _rat['trials'][rec_day]['events']_: Global time of the first lever press in the trial (in seconds).

> _rat['trials'][rec_day]['times']_: Time-within-trial corresponding to the stored behavioral data.

> _rat['trials'][rec_day]['kinematics_w']_: Dictionary with time-warped position data for each forelimb ('paw_L'/'paw_R'). Each entry contains an array of shape (trials x timepoint x coordinate).

> _rat['trials'][rec_day]['vels_w']_: Dictionary with first derivative of the position data.

> _rat['trials'][rec_day]['acc_w']_: Dictionary with second derivative of the position data.


_rat['units']_: neural data for each recorded unit (rat['units'][unum]) and recording day (rat['units'][unum][rec_day]). Note that there can be gaps between days if the unit was not recorded or there was a spike sorting error on a given day.

> _rat['units'][unum][rec_day]['raster_w']_: List of time-warped spiketimes for each trial.

> _rat['units'][unum][rec_day]['keep']_: List of whether the unit passed quality control on the corresponding trial (QC performed session-wise for the two sessions on each day). No spike times are available for trials indicated by 'False', which should not be used for analyses.

> _rat['units'][unum][rec_day]['peth_w']_: Time-warped PETH for the unit across all trials on this day after Gaussian convolution.

> _rat['units'][unum][rec_day]['peth_w_t']_: Trial-wise time-warped PETH for the unit (shape: trials x timebins) in the absence of convolution.

_rat['name']_: name of animal

_rat['unittypes']_: putative unit types (1 for projection neurons, 2 for interneurons).

### Wet-dog shake data

To load the WDS data, use load_rat('Hindol_wds') and similar. This data has a few differences from the lever-pressing data. Most notably,
> _rat['trials'][rec_day]['kinematics_w']_ has only a single entry ('acc') which contains the acceleration in the x, y, and z directions and is identical to 'vels_w' and 'acc_w'.

> _rat['trials'][rec_day]['events']_ contains the time of the first peak of the accelerometer trace.




