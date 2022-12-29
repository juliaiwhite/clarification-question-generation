# Clarification Question Generation

To use this repo...
- Create a conda environment with `conda env create -f environment.yml`.
- Activate the conda environment with `conda activate clarification-question-generation`.
- Clone the transformer-based image captioning model at [krasserm/fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning/blob/master/README.md) with `git clone https://github.com/krasserm/fairseq-image-captioning.git`. Then follow the instructions in the [krasserm/fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning/blob/master/README.md) repo to download the MS-COCO dataset and the pre-trained baseline model [checkpoint 24](https://martin-krasser.de/image-captioning/checkpoint24.pt).
- Train a response classifier with `python train_response_model.py`.
- Run the question asking model with `python run_question_asking_model.py`. Results will be saved to `outputs/`.
