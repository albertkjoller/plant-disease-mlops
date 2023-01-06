# -*- coding: utf-8 -*-
import os
import glob
from pathlib import Path

import click
import logging
from dotenv import find_dotenv, load_dotenv

import torch
from torchvision.io import read_image
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

from plantvillage import PlantVillage
from create_loaders import Rescale

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    input_filepath, output_filepath = Path(input_filepath), Path(output_filepath)

    plantvillage_data = ImageFolder(input_filepath, transform=transforms.Compose([
        transforms.Resize(128),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), 
        transforms.ToTensor()],
        ),
    )

    # Size of data
    N = len(plantvillage_data)

    # Size of train, test and validation splits
    N_test = int(round(N * 0.2, 0))
    N_trainval = N - N_test
    N_val = int(round(N_trainval * 0.2, 0))
    N_train = N_trainval - N_val

    # Split dataset into files
    trainval_data, val_data, test_data = torch.utils.data.random_split(plantvillage_data, [N_train, N_val, N_test])
    train_data, val_data = torch.utils.data.random_split(trainval_data, [N_train, N_val])
    
    train_dl = DataLoader(dataset, bs, sampler=torch.utils.data.SubsetRandomSampler(indices[:300]))
    test_dl  = DataLoader(dataset, bs, sampler=torch.utils.data.SubsetRandomSampler(indices[-300:]))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=4
                                               )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()