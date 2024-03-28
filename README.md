## Introduction
The historical reviews of users about items are often insufficient, thus making it difficult to ensure the accuracy of generated explanation text. To address this issue, we propose a novel model-ERRA  with retrieval enhancement. With this additional information, we can generate more accurate and informative explanations.
## Usage
Below are examples of how to run ERRA (with and without the retrieval enhancement).
python -u main.py \
--data_path ./reviews.json \
--cuda 1
--checkpoint ./tripadvisorf/ \
--attention_mask >> 1
--use_retrieval >> 1

Our data reprocessing is based on [https://lifehkbueduhk-my.sharepoint.com/personal/16484134_life_hkbu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F16484134%5Flife%5Fhkbu%5Fedu%5Fhk%2FDocuments%2FCIKM20%2DNETE%2DDatasets&ga=1](https://lifehkbueduhk-my.sharepoint.com/personal/16484134_life_hkbu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F16484134%5Flife%5Fhkbu%5Fedu%5Fhk%2FDocuments%2FCIKM20%2DNETE%2DDatasets&ga=1)
## Code dependencies
- Python== 3.7
-	PyTorch ==1.12.1
-	transformers==4.25.1
-	pandas==1.4.3
-	mkl-service==2.4.0
-	nltk==3.7
-	tokenizers==0.13.2
-	ply==3.11
## PyTorch implementation of ERRA model 
- main.py is used for train a ERRA model.
- module.py is the construction and details of the model.
- utils.py has functions for processing input files.
- BLUE.py folder contains the tool and a example script of text evaluation.



