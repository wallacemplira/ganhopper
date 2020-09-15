## GANHopper: Multi-Hop GAN for Unsupervised Image-to-Image Translation

[Link for the paper published on Arxiv](https://arxiv.org/abs/2002.10102)

### News

* GANHopper was presented at ECCV 2020. See presentation [here](https://www.youtube.com/watch?v=jQ-dxwgBm3Q&t=8s&ab_channel=WM).

### Instrunctions

To run the code, download the datasets into the 'datasets' folder. The internal structure of each dataset should follow the one in datasets/example.

To run the evaluation on this dataset (or on any of the original CycleGAN datasets), please just run
```
python main.py --dataset_dir={{dataset_dir}}
```

Where {dataset_dir} is a folder containing a dataset in the format described above. The dog_cat_faces dataset can be downloaded at https://github.com/brownvc/ganimorph

To reproduce the results from the dog_cat_faces dataset please use the default parameters. To reproduce our results on human2cats, add to the command instruction the parameters *--epoch_step=22* and *--epoch=22*.
Similarly, to reproduce our results on human2dolls, add to the command instruction the parameters *--epoch_step=25* and *--epoch=25*.


The objective function parameters can be controlled with can be set with the following arguments:
               --hybridness: weight of the hybridness loss
               --h_hops: total number of translation hops between two domains
               --smootheness: weight of the smootheness loss
               --L1_lambda: weight on reconstruction loss term between hops in objective
               --adversarial: weight of the adversarial loss

However, the experiments mentioned above use the default values for all hyperparameters.

### Dependencies
```
tensorflow-gpu=1.9.0
numpy=1.15.2
scipy=1.1.0
pillow=3.3.0
imageio=2.4.1
```
