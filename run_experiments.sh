#!/bin/bash

# for i in {1..10}; do python main.py >> results.txt; done
# SET SEED
# python process_news.py

#python main_cs_pz.py --beta=0 --gamma=0 --setting="mimic" --num_epochs=600
#
# python main_hsic.py --setting="mimic" --num_epochs=600 
# python main_giks.py --setting="mimic" --num_epochs=600 
# python main_wass.py --setting="mimic" --num_epochs=600 
python main_cs_pz.py --setting="mimic" --num_epochs=600 --gamma=0.001 --beta=0.1 # <- best performing
python main_cs_pz.py --setting="mimic" --num_epochs=600 --gamma=0.00 --beta=0.1 # <- best performing
python main_cs_pz.py --setting="mimic" --num_epochs=600 --gamma=0.001 --beta=0 # <- best performing