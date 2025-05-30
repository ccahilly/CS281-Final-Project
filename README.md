# Explaining Rider vs. Person Classification for Detectron2 Cityscapes Mask RNN

## About [Detectron 2](https://github.com/facebookresearch/detectron2)

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>
<br>

## Set Up
First, clone this repo, set up the conda environment, and install detectron2:
```
git clone git@github.com:ccahilly/CS281-Final-Project.git
cd CS281-Final-Project
conda env create -f detectron_environment.yml -n detectron
conda activate detectron
python -m pip install -e .
```

Lastly, [download](https://www.cityscapes-dataset.com/downloads/) the Cityscapes dataset (specifically, `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`). Place `leftImg8bit` and `gtFine` in `datasets/cityscapes`.

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
