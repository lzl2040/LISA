## Train
merget weights:
```
conda activate llava
python3 -m llava.model.apply_delta \
    --base huggyllama/llama-7b \
    --target /home/v-zuoleili/Pretrain/LLaVA-7B-v0 \
    --delta liuhaotian/LLaVA-7b-delta-v0
```
```shell
deepspeed --master_port=24999 train_ds.py \
  --version="/home/v-zuoleili/Pretrain/LLaVA-7B-v0" \
  --dataset_dir='/home/v-zuoleili/Data' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="refer_seg" \
  --sample_rates="1" \
  --exp_name="lisa-7b"
```