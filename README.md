## Requirements
- numpy
- cv2
- matplotlib
- scipy
- argparse
- matplotlib==3.4.2
- tqdm==4.61.2

## Project Structure
- project2/
    - code/
        - main.py
        - myutils.py
        - pr3_utils.py
        - slam_scratch_code.ipynb (irrelevant)
    - data/
        - 03.npz
        - 10.npz
    - plots/
        - contains all plots

## To run the particle filter SLAM code
```
cd project3/code/
```
```
python3 main.py --ds=3 --mapping --featSkip=6 --initPoseCov=0.01 --initLMCov=1.0 --vcov=10 --wcov=1e-2 --distThresh=200
```
- `--ds` (either 3 or 10)
- `--mapping` if specified, then performs visual mapping otherwise directly performs visual-inertial SLAM
- `--initPoseCov`: initial pose covariance diagonal values
- `--initLMCov`: initial LM covariance diagonal values
- `--vcov`: observation model noise covariance values
- For all parts (IMU localization, visual mapping and visual-inertial SLAM) - The main code is `main.py` while most utility functions are implemented in `myutils.py` and in `pr3_utils.py`
