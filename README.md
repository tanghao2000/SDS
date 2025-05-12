# SDS
## Reqirements
### create conda env
conda create -n SDS python=3.9

### install packages
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101

pip install opencv-python ftfy regex tqdm ttach tensorboard lxml cython

### install pydensecrf from source
git clone https://github.com/lucasb-eyer/pydensecrf

cd pydensecrf

python setup.py install

## Preparing Datasets
### PASCAL VOC2012
### MS COCO2017
### Preparing pre-trained model
Download CLIP pre-trained [ViT-B/16] at and put it to /your_home_dir/pretrained_models/clip.

## Run Experiments
### Step 1. Perturbation-based CLIP Similarity (PCS)
For VOC

`CUDA_VISIBLE_DEVICES=0 python PCS_voc.py --img_root /your_home_dir/datasets/gen_voc/image --split_file ./voc/train.txt --model /your_home_dir/pretrained_models/clip/ViT-B-16.pt --num_workers 1 --cam_out_dir ./output/voc/cams`

For COCO

`CUDA_VISIBLE_DEVICES=0 python PCS_coco.py --img_root /your_home_dir/datasets/gen_coco/image --split_file ./coco/train.txt --model /your_home_dir/pretrained_models/clip/ViT-B-16.pt --num_workers 1 --cam_out_dir ./output/coco/cams`


### Step 2. Generate Pesudo Annotation
For VOC

`python generate_pesudo_annotation.py --cam_out_dir ./output/voc/cams --gt_root /your_home_dir/datasets/gen_voc/mask --image_root /your_home_dir/datasets/gen_voc/image --split_file ./voc/train.txt --pseudo_mask_save_path ./output/voc/pseudo_annotation`

For COCO

`python generate_pesudo_annotation.py --cam_out_dir ./output/coco/cams --gt_root /your_home_dir/datasets/gen_coco/mask --image_root /your_home_dir/datasets/gen_coco/image --split_file ./coco/train.txt --pseudo_mask_save_path ./output/coco/pseudo_annotation`

### Step 3. Annotation Similarity Filter (ASF)
With input the mask of gen_voc and gen_coco

`CUDA_VISIBLE_DEVICES=0 python ASF.py`


### Step 4. Train Segmentation Model
We select DeeplabV3, Mask2Former and CDL as the segmentation models.
