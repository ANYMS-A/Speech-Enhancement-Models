# Speech enhancement models using spectrograms as features

# Speech-Enhancement-Models
Speech enhancement models:MLP, Auto-encoder, GAN

## Dataset
The dataset is the speech enhancment dataset built by the University of Edinburgh.
[DataShare](https://datashare.is.ed.ac.uk/handle/10283/2791).

## Requirements
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* librosa
```
pip install librosa
```
## Notes:
The audios should be sliced into pieces with equal time length. Then do Short Time Fourier Transform on them, turn them into a 2D matirx. Then we use CNN to extract features from them.
