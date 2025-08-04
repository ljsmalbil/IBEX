#!/bin/bash

# for i in {1..10}; do python main.py >> results.txt; done
# SET SEED
# python process_news.py

python main_cs_pz.py --setting="news" --num_epochs=600 --gamma=0.001 --beta=0.1 # <- best performing
