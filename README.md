![image](https://github.com/user-attachments/assets/53b51e53-bf2b-42d9-bd40-7f87bd004e3d)

This is the official repository of the paper "RetFiner: A Vision-Language Refinement Scheme for Retinal Foundation Models", by Ronald Fecso, José Morano, Ursula Schmidt-Erfurth, and Hrvoje Bogunović, accepted for presentation at [MICCAI 2025](https://conferences.miccai.org/2025/en/).

#### [[`MICCAI paper`](https://papers.miccai.org/miccai-2025/paper/0660_paper.pdf)]

## Overview

We propose RetFiner (Fig. 1), an SSL vision-language refinement scheme that improves the representations of existing FMs and enables their efficient and direct adaptation to specific populations for improved downstream performance. Our method uses a diverse set of training objectives which take advantage of the rich supervisory signal found in textual data. We tested RetFiner on the retinal FMs RETFound, UrFound, and VisionFM (Table 1), showing significant improvements in linear probing performance on seven highly diverse OCT classification tasks, with an average increase of 5.7, 3.9, and 2.1 percentage points over their baselines, respectively .

<br>

![image](https://github.com/user-attachments/assets/20f171e3-a3b9-4e97-af1b-c86008a20925)

<div align="center">
<b>Fig. 1. RetFiner method.</b> Cross-attention layers are activated only during the forward passes for ITM, MLM, and GM. An example of an OCT image and report is shown.
</div>

<br>

<div align="center">
<b>Table 1. Average linear probing performance over all downstream datasets.</b> The best metric value out of all the models (<b>bolded</b>) and out of the base models (<ins>underlined</ins>) were compared to measure if there was a statistically significant difference using the Wilcoxon signed-rank test (**: p < 0.01, ***: p < 0.001). Values in parentheses represent change in performance compared to their baseline counterpart.
</div>

![image](https://github.com/user-attachments/assets/69a778f0-21cb-46ec-a8c5-1e2c52c04de3)


## Model weights

The model weights are available in the [Model weights release](https://github.com/ronnief1/RetFiner/releases/tag/weights) on GitHub.

| Model | Link |
| --- | --- |
| RetFiner-RETFound | [Weights-RetFiner-R](https://github.com/ronnief1/RetFiner/releases/download/weights/RetFiner_RETFound.pth) |
| RetFiner-UrFound | [Weights-RetFiner-U](https://github.com/ronnief1/RetFiner/releases/download/weights/RetFiner_UrFound.pth) |
| RetFiner-VisionFM | [Weights-RetFiner-V](https://github.com/ronnief1/RetFiner/releases/download/weights/RetFiner_VisionFM.pth) |


## RetFining
If you want to run RetFiner on your vision model:

Navigate into RetFiner/

Create a new virtual environment in RetFiner/ and install requirements.txt

Text encoder weights: Download BERT [model](https://github.com/ronnief1/RetFiner/releases/download/weights/bert-base-uncased-crossattn.zip) and [tokenizer](https://github.com/ronnief1/RetFiner/releases/download/weights/tokenizer.zip) and unzip them RetFiner/pretrained_weights/.

Vision encoder weights: Put your vision model in RetFiner/

Our in-house image-text training data is private so you will need to use your own. Edit the dataloader in RetFiner/ImageCaptionDataset.py accordingly. __getitem__ should return a list consisting of two elements: an image (torch tensor) and a report (string).

Then in the command line run:
```sh
python train.py --model_weights path/to/yourvisionmodel
```

Once your model is trained, run the following script to extract the vision backbone. This will save it under ../linear_probing/_weights. Note this has only been tested on RETFound, VisionFM, and our in-house MAE. You may need to alter it for another FM.
```sh
python get_vision_backbone_for_linprobing.py --path_to_model models/<model name>/best-model.ckpt
```

## Linear probing

Once you have your RetFined model, navigate into ../linear_probing/, set up a new virtual environment there, and then activate it. Then install requirements.txt.

Then you can run one of the .sh scripts based on which model you have.

For example, in retfound.sh, you would change the ft_weights arg to _weights/<my_model_name>. Adjust the data sets arg accordingly.

Results are found in __results/.

## Linear probing datasets

- Duke iAMD: https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm
- Harvard Glaucoma: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP
- Noor Eye Hospital: https://hrabbani.site123.me/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme
- OCTDL: https://data.mendeley.com/datasets/sncdhf53xc/4
- OCTID: https://borealisdata.ca/dataverse/OCTID
- NEHUT: https://data.mendeley.com/datasets/8kt969dhx6/1



## License

The models and associated code are released under the CC-BY-NC 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. See [LICENSE](LICENSE) for more details.

Specifically, the following licenses apply to the base models and fine-tuned models:


| Model Name      | Base Model | Original License | Fine-Tuned License |
|-----------------|------------|------------------|---------------------|
| **RetFiner-U**  | [UrFound](https://github.com/yukkai/UrFound) | [MIT](https://opensource.org/licenses/MIT) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| **RetFiner-R**  | [RETFound](https://github.com/rmaphoh/RETFound_MAE) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| **RetFiner-V**  | [VisionFM](https://github.com/ABILab-CUHK/VisionFM) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |



## Citation

If you use any of our models, please do the following:

1. **Cite the original base models**:
   - **UrFound**: Yu, Kai, et al. "UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2024.
   - **RETFound**: Zhou, Yukun, et al. "A foundation model for generalizable disease detection from retinal images." Nature 622.7981 (2023): 156-163.
   - **VisionFM**: Qiu, Jianing, et al. "Development and validation of a multimodal multitask vision foundation model for generalist ophthalmic artificial intelligence." NEJM AI 1.12 (2024): AIoa2300221.
2. **Cite this work**:
   ```bibtex
   @InProceedings{FecRon_RetFiner_MICCAI2025,
        author = { Fecso, Ronald and Morano, José and Schmidt-Erfurth, Ursula and Bogunović, Hrvoje},
        title = { { RetFiner: A Vision-Language Refinement Scheme for Retinal Foundation Models } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15964},
        month = {September},
        page = {543 -- 553}
   }
   ```
