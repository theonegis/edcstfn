# An Enhanced Deep Convolutional SpatioTemporal Network for Remote Sensing Images (EDCSTFN)

**Notice:** 

- The source code has been refactored. If this causes any runtime error, please feel free to contact me or make a pull request.
- There is a small mistake in Figure. 2 regarding the original manuscript, the input and out channels should be the same for the AutoEncoder. (I wrote to the editor, but they said it cannot be corrected after the paper being released.) 

**Models:**

- The master branch is the implementation of the EDCSTFN model.

- The autoencoder branch is the implementation of the hourglass AutoEncoder. The pretrained Encoder subnet is used in the EDCSTFN to calculate the feature loss.

**Environment:**

Tested on the following environment:

- Python: >=3.6

- PyTorch: >=0.4

If you find this code useful in your research, please consider citing our work:

```
@Article{rs11242898,
AUTHOR = {Tan, Zhenyu and Di, Liping and Zhang, Mingda and Guo, Liying and Gao, Meiling},
TITLE = {An Enhanced Deep Convolutional Model for Spatiotemporal Image Fusion},
JOURNAL = {Remote Sensing},
VOLUME = {11},
YEAR = {2019},
NUMBER = {24},
ARTICLE-NUMBER = {2898},
URL = {https://www.mdpi.com/2072-4292/11/24/2898},
ISSN = {2072-4292},
DOI = {10.3390/rs11242898}
}
```

https://www.mdpi.com/2072-4292/11/24/2898/htm
