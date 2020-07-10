# Fast Video BLIINDS -- Python

This is a fast python implementation of Video - BLIINDs, a blind natural video quality prediction algorithm, proposed [HERE](https://live.ece.utexas.edu/publications/2014/VideoBLIINDS.pdf)

The code is optimized using Numba python package. Other python package dependencies are listed in [requirements.txt](requirements.txt)

## Usage
Run the following with path to the video as argument
```
python3 vbliinds_demo.py --vid_path test.mp4
```

[1] M. Saad and A.C. Bovik, "Blind prediction of natural video quality," IEEE Transactions on Image Processing , December, 2013.
