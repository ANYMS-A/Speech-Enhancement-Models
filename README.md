# Speech enhancement models using spectrograms as features

# Speech-Enhancement-Models
Speech enhancement models:MLP, Auto-encoder, GAN

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
