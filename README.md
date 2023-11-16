# MORSE
This repo contains the code for the S&P 23 paper titled "From Grim Reality to Practical Solution: Malware Classification in Real-World Noise".

## Requirement
This codebase is written for ```python3.7```. The requirement of our method is listed in ```requirements.txt```.

## Code Structure and instructions
### Basics
- Our method is implemented in the file `our_match.py`.

### Data
- You could download the dataset from google drive https://tinyurl.com/skvw9n7j (i.e., include both the synthetic dataset and real-world PE malware dataset) and put them in the `data` folder.

## Training
- Code for training MORSE is in the following file: `run_ourmatch.py`.
```
usage: run_ourmatch.py [--lr learning rate] [--batch_size batch_size] [--input_dim input_dim] [--epoch epoch] [--warmup warmup_epoch]
                       [--noise_rate noise_rate] [--noise_type noise_type] [--imb_type imb_type] [--imb_ratio imb_ratio] [--threshold threshold]
                       [--reweight_start reweight_start_epoch] [--dataset_origin dataset]

arguments:
  --lr               learning rate (default value is the value in the file `run_ourmatch.py`)
  --batch_size       batch size (default value is the value in the file `run_ourmatch.py`)
  --input_dim        extracted malware feature dimension (i.e., 2381 in the synthetic dataset)
  --epoch            total training epochs
  --warmup           warmup period (i.e., 10 in the synthetic dataset and 5 in the PE malware dataset)
  --noise_rate       noise ratio of the dataset
  --noise_type       noise type of the dataset
  --imb_type         imbalance type of the dataset (either none or step)
  --imb_ratio        imbalance ratio of the dataset (i.e., 0.05/0.01 in the synthetic dataset)
  --threshold        pseudo label threshold (i.e., 0.40 in the synthetic dataset and 0.95 in the PE malware dataset)
  --reweight_start   starting reweighting epoch
  --dataset_origin   dataset for training (i.e., either the synthetic or real-world dataset)
```
