## Train
merget weights:
```
git clone https://github.com/haotian-liu/LLaVA.git
conda create -n llava python=3.10
conda activate llava
cd LLaVA
pip install -e .
pip install protobuf
python3 -m llava.model.apply_delta \
    --base huggyllama/llama-7b \
    --target /home/v-zuoleili/Pretrain/LLaVA-7B-v0 \
    --delta liuhaotian/LLaVA-7b-delta-v0
```
```shell
deepspeed --master_port=24999 train_ds.py \
  --version="/home/v-zuoleili/Pretrain/LLaVA-7B-v0" \
  --dataset_dir='/home/v-zuoleili/Data' \
  --vision_pretrained="/home/v-zuoleili/Pretrain/sam_vit_h_4b8939.pth" \
  --dataset="refer_seg" \
  --sample_rates="1" \
  --exp_name="lisa-7b" \
  --image_size=480
```