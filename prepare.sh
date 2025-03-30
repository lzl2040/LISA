# download data
sudo apt install unzip
mkdir /home/v-zuoleili/Data
cd /home/v-zuoleili/Data
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d ./ && rm train2014.zip
wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
unzip refcoco.zip && rm refcoco.zip
unzip refcoco+.zip && rm refcoco+.zip
unzip refcocog.zip && rm refcocog.zip

# create env
# cd /home/v-zuoleili
# wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
# bash Anaconda3-2024.10-1-Linux-x86_64.sh
# conda create -n ris python=3.10
# conda activate ris
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# pip install -r requirements.txt
