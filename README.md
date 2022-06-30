Code of ICMR 2022 Paper:

Local Slot Attention for Vision-and-Language Navigation
Yifeng Zhuang, Qiang sun, Yanwei Fu, Lifeng Chen, Xiangyang Xue

## Prerequisites

Please following the **Prerequisites** chapter from [Recurrent-VLN-Bert](https://github.com/YicongHong/Recurrent-VLN-BERT), than replace the `r2r_src` and `run` folder with ones in our repo.

## Pre-trained Weights

Will be available soon.

## Maxpooling End-feature

Download the `ResNet-152-places365-maxpool.pkl` from [Google Drive](https://drive.google.com/file/d/1KpRNmtdeRzQxMc2hsOWfc2C8w6yMI2WK/view?usp=sharing) and place it in `img_features` folder.

## Run

`bash run/train_agent_slot.bash`

## Test

`bash run/test_agent.bash`
