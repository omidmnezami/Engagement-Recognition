# Engagement Recognition

TensorFlow implementation of [Engagement Recognition using Deep Learning and Facial Expression](https://arxiv.org/abs/1808.02324) proposing a deep learning model to recognize engagement from human faces.

<p align="center">
<img src="images/sample_eng.jpg">
</p>

### References
Please cite our paper if you use our code or model:
```
@article{nezami2018deep,
  title={Engagement Recognition using Deep Learning and Facial Expression},
  author={Nezami, Omid Mohamad and Hamey, Len and Richards, Deborah and Dras, Mark and Wan, Stephen and Paris, Cecile},
  journal={arXiv preprint arXiv:1808.02324},
  year={2018}
}

```
### Requiremens

```
Python 2.7.12
Tensorflow 1.8.0

```

### Contents

1. [CNN Model Source Code](/code/CNN_model.py)
2. [VGG Model Source Code](/code/VGG_model.py)
3. [Transfer Model Source Code](/code/VGG_model.py)


### Test
1. Download [trained model](/model/TF_final.data-00000-of-00001), and put it under `model\`.
2. Run the model's script:
    python VGG_model.py test
    
    
### Training
1. Dowload [pretrained model](/model/TF_start.data-00000-of-00001), and put it under 'model\'
2. Run the model's script:
    python VGG_model.py train
    
