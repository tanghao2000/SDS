# SDS
Reqirements
# create conda env
conda create -n SDS python=3.9

# install packages
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101
pip install opencv-python ftfy regex tqdm ttach tensorboard lxml cython

# install pydensecrf from source
git clone https://github.com/lucasb-eyer/pydensecrf
cd pydensecrf
python setup.py install
