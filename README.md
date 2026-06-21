# M^2CoFS
Codes for ***Breaking Task Boundaries: A Unified Model for 3D Medical Image Fusion and Segmentation Guided by Manifold Perspective.*** ***(AAAI 2026 Oral Presentation)***
[Paper](https://doi.org/10.1609/aaai.v40i12.38008)

------

![pipline](pipeline.png)

------


### 1. Recommended Environment:
 - [ ] python = 3.8
 - [ ] torch = 1.12.1
 - [ ] monai = 1.1.0
 - [ ] numpy = 1.24.2
 - [ ] SimmpleITK = 2.2.1
 - [ ] pillow = 9.4.0
 - [ ] scikit-image = 0.19.3
 - [ ] imgaug = 0.4.0
 - [ ] trochio = 0.18

---

### 2. Pre-trained Weights

The pre-trained [weights] are available at:

[Google Drive](https://drive.google.com/drive/folders/1UcnNtUpYIns2Bv7Ya6yTggGITcBhcA51)

Please put the downloaded weights in `./train_result/`.

---

### 3. Test Data

The test data can be downloaded from the following [website]:

[Dataset Website](https://mrbrains13.isi.uu.nl/index.html) 

Please put the processed test data in `./test_img/`.

The test data should be organized as follows:

```text
test_img/
├── whole/
│   ├── t1/
│   │   └── ...
│   ├── t2-flair/
│   │   └── ...
│   └── label/
│       └── ...
└── patches/
    ├── t1/
    │   └── ...
    └── t2-flair/
        └── ...
```

The `whole` folder is used for image fusion testing, and the `patches` folder is used for segmentation testing.

---

### 4. Test:
* Prepare test data: put the processed data in './test_img'<br>
* Run ```python test.py```<br>
---

### 5. Citation

If this code is useful for your research, please cite our paper:

```bibtex
@inproceedings{wang2026breaking,
  title={Breaking Task Boundaries: A Unified Model for 3D Medical Image Fusion and Segmentation Guided by Manifold Perspective},
  author={Wang, Zeyu and Wang, Jiayu and Song, Haiyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={12},
  pages={10376--10384},
  year={2026}
}
```
