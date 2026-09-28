cd submodules/Mask2Former
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl
cd ../..
