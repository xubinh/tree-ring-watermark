#!/usr/bin/env bash

python run_tree_ring_watermark_remake.py --run_name rotation --w_channel 3 --w_pattern ring --r_degree 75 --start 0 --end 100 --with_tracking
# python run_tree_ring_watermark_remake.py --run_name jpeg --w_channel 3 --w_pattern ring --jpeg_ratio 25 --start 0 --end 100 --with_tracking
# python run_tree_ring_watermark_remake.py --run_name cropping --w_channel 3 --w_pattern ring --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end 100 --with_tracking
# python run_tree_ring_watermark_remake.py --run_name blurring --w_channel 3 --w_pattern ring --gaussian_blur_r 4 --start 0 --end 100 --with_tracking
# python run_tree_ring_watermark_remake.py --run_name noise --w_channel 3 --w_pattern ring --gaussian_std 0.1 --start 0 --end 100 --with_tracking
# python run_tree_ring_watermark_remake.py --run_name color_jitter --w_channel 3 --w_pattern ring --brightness_factor 6 --start 0 --end 100 --with_tracking
