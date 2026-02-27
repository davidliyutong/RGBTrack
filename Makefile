SHELL := /bin/bash
.ONESHELL:

calibrate:
	source ~/miniconda3/etc/profile.d/conda.sh
	conda activate RGBTrack
	python -u calibrate.py
up:
	source ~/miniconda3/etc/profile.d/conda.sh
	conda activate RGBTrack
	python -u main.py