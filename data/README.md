# Data

We use the public **Brain Tumor MRI Images – 44 Classes** dataset (4,479 T1, T1C+ and T2 images). The images are not included in this repo.

Download the dataset and unzip it here so the layout is:

```
data/44_classes/
├── Training/
└── Testing/
```

The notebooks were run on Google Colab and download the data themselves with `gdown` into `44_classes/`. If you run them locally, change the paths `44_classes/Training` and `44_classes/Testing` to `data/44_classes/...`.
