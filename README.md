# 🧠 SynthStroke: Deep Learning Stroke Lesion Segmentation

Python implementation of "Synthetic Data for Robust Stroke Segmentation"

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/synthstroke.git
   cd synthstroke
   ```

2. Create a conda environment (recommended):
   ```bash
   conda create -n synthstroke python=3.10
   conda activate synthstroke
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Command Line Interface
```bash
# Training baseline model
python train.py --name baseline_model \ # model name for folder + wandb
                --logdir ./ \ # parent folder to write experiments. will create new folder {logdir}/{name}/
                --baseline \ # tell training script to use the baseline dataloader (i.e. real images)
                --l2 50 \ # train using L2 loss for first 50 epochs
                --patch 128 \ # patch size for random crops in training
                --amp \ # auto-mixed precision - should make training faster + allow larger batch size
                --epochs 500 \ # number of epochs
                --epoch_length 200 \ # number of iters per epoch
                --lr 0.001 \ # initial learning rate
                --val_interval 2 \ # save weights and calculate val dice every 2 epochs

# Train fancy Synth model
python train.py --name fancy_synth_model \ # model name for folder + wandb
                --logdir ./ \ # parent folder to write experiments. will create new folder {logdir}/{name}/
                --mbhealthy \ # tell training script to use MultiBrain healthy labels
                --fade \ # use INU fields to mimic penumbra within lesion masks
                --lesion_weight 2 \ # upweight lesion class by 2 relative to healthy tissue in seg loss
                --l2 50 \ # train using L2 loss for first 50 epochs
                --patch 128 \ # patch size for random crops in training
                --amp \ # auto-mixed precision - should make training faster + allow larger batch size
                --epochs 500 \ # number of epochs
                --epoch_length 200 \ # number of iters per epoch
                --lr 0.001 \ # initial learning rate
                --val_interval 2 \ # save weights and calculate val dice every 2 epochs

# Test (infer) fancy Synth model on new data - this does not require ground truth data
python test.py --weights ./fancy_synth_model/checkpoint.pt \ # path to our model trained in previous script
               --tta \ # use test-time augmentation when predicting
               --mb \ # predict multi-brain labels (this determines the out channels in the model so please set this if used in training, even if you don't want healthy labels)
               --patch 128 \ # same size we used in training - will use Sliding Window Inferer
               --savedir /my/output/data/folder/ \ # folder to write predictions to
               --files /my/input/data/folder/*.nii.gz \ # regex to generate list of files. can also be a path to a .txt containing a single column of file paths
```

## 🏋️ Weights

Pre-trained model weights will be available for download soon. Please check back later or watch this repository for updates.

The weights will include:
- Baseline model trained on real stroke data
- SynthStroke model trained with synthetic data augmentation

## 🆘 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## 📚 Citation

If you use SynthStroke in your research, please cite:

Chalcroft, L., Pappas, I., Price, C. J., & Ashburner, J. (2024). Synthetic Data for Robust Stroke Segmentation. arXiv preprint arXiv:2404.01946. https://arxiv.org/abs/2404.01946

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
