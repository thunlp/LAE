# Fine-Grained Legal Argument-Pair Extraction via Coarse-Grained Pre-training (COLING 2024)

The code of our paper "Fine-Grained Legal Argument-Pair Extraction via Coarse-Grained Pre-training".

## Installation

```bash
pip install -r requirements.txt
```

## Pre-training

First, download the pre-trained law BERT model from [here](https://thunlp-public.oss-cn-hongkong.aliyuncs.com/legal/LegalArgumentPairExtraction/law_bert.zip) and save it to `./pretrained_models/law_bert`.

Second, download the LAE dataset from [here](https://thunlp-public.oss-cn-hongkong.aliyuncs.com/legal/LegalArgumentPairExtraction/data.zip) and save it to `./data/lae`.

Then, run the following command to pre-train the model on the LAE dataset:

```bash
mpirun --allow-run-as-root -n 8 python run_mindformer.py --config configs/txtcls/contract_train.yaml # for contract domain
mpirun --allow-run-as-root -n 8 python run_mindformer.py --config configs/txtcls/loan_train.yaml # for loan domain
```

Finally, run the following command to evaluate the pre-trained model on the LAE dataset:

```bash
# for contract domain
for file in $(ls output/checkpoint/rank_0/contract_pretrain_rank_0-*.ckpt); do
    echo "Processing $file"
    python run_mindformer.py --config configs/txtcls/contract_test.yaml --load_checkpoint $file
done
# for loan domain
for file in $(ls output/checkpoint/rank_0/loan_pretrain_rank_0-*.ckpt); do
    echo "Processing $file"
    python run_mindformer.py --config configs/txtcls/loan_test.yaml --load_checkpoint $file
done
```

It will iterate over all the checkpoints and evaluate them on the LAE dataset. If you rerun the pre-training, you need to first remove previous checkpoints to ensure the evaluation is correct. After the evaluation, you can select the best checkpoint based on the evaluation results and use it for fine-tuning.

## Fine-tuning

First, run the following command to fine-tune the pre-trained model on the target dataset:

```bash
python run_mindformer.py --config configs/txtcls/contract_finetune.yaml --load_checkpoint output/checkpoint/rank_0/contract_pretrain_rank_0-{best_checkpoint}.ckpt # for contract domain
python run_mindformer.py --config configs/txtcls/loan_finetune.yaml --load_checkpoint output/checkpoint/rank_0/loan_pretrain_rank_0-{best_checkpoint}.ckpt # for loan domain
```

Here, `{best_checkpoint}` is the best checkpoint selected from the pre-training evaluation.

Finally, run the following command to evaluate the fine-tuned model on the target dataset:

```bash
# for contract domain
for file in $(ls output/checkpoint/rank_0/contract_finetune_rank_0-*.ckpt); do
    echo "Processing $file"
    python run_mindformer.py --config configs/txtcls/contract_test.yaml --load_checkpoint $file
done
# for loan domain
for file in $(ls output/checkpoint/rank_0/loan_finetune_rank_0-*.ckpt); do
    echo "Processing $file"
    python run_mindformer.py --config configs/txtcls/loan_test.yaml --load_checkpoint $file
done
```

Note that in `contract_test.yaml` and `loan_test.yaml`, we specify the development dataset for evaluation. You can change it to test dataset for final evaluation. Besides, if you rerun the fine-tuning, you also need to remove previous checkpoints to ensure the evaluation is correct. 


## Citation
```bibtex
@inproceedings{xiao2024fine,
    title = {Fine-Grained Legal Argument-Pair Extraction via Coarse-Grained Pre-training},
    author = {Chaojun Xiao and Yutao Sun and Yuan Yao and Xu Han and Wenbin Zhang and Zhiyuan Liu and Maosong Sun},
    booktitle = {Proceedings of the 30th International Conference on Computational Linguistics (COLING 2024)},
    year = {2024},
}
```