#!/bin/bash

python pca_learn_dist.py -t resnet101_gem --imsize 512 --embed_dim 2048 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_gem_pca_2048d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_gem --imsize 512 --embed_dim 1024 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_gem_pca_1024d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_gem --imsize 512 --embed_dim 256 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_gem_pca_256d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_gem --imsize 512 --embed_dim 128 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_gem_pca_128d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_gem --imsize 512 --embed_dim 64 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_gem_pca_64d_gldv2_512x512_p3_randcrop.pt

python pca_learn_dist.py -t resnet101_ap_gem --imsize 512 --embed_dim 2048 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_ap_gem_pca_2048d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_ap_gem --imsize 512 --embed_dim 1024 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_ap_gem_pca_1024d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_ap_gem --imsize 512 --embed_dim 256 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_ap_gem_pca_256d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_ap_gem --imsize 512 --embed_dim 128 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_ap_gem_pca_128d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_ap_gem --imsize 512 --embed_dim 64 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_ap_gem_pca_64d_gldv2_512x512_p3_randcrop.pt

python pca_learn_dist.py -t resnet101_solar --imsize 512 --embed_dim 2048 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_solar_pca_2048d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_solar --imsize 512 --embed_dim 1024 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_solar_pca_1024d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_solar --imsize 512 --embed_dim 256 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_solar_pca_256d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_solar --imsize 512 --embed_dim 128 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_solar_pca_128d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_solar --imsize 512 --embed_dim 64 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_solar_pca_64d_gldv2_512x512_p3_randcrop.pt

python pca_learn_dist.py -t resnet101_delg --imsize 512 --embed_dim 256 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_delg_pca_256d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_delg --imsize 512 --embed_dim 128 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_delg_pca_128d_gldv2_512x512_p3_randcrop.pt
python pca_learn_dist.py -t resnet101_delg --imsize 512 --embed_dim 64 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_delg_pca_64d_gldv2_512x512_p3_randcrop.pt

python pca_learn_dist.py -t resnet101_dolg --imsize 512 --embed_dim 256 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_dolg_pca_256d_gldv2_512x512_p3_global_randcrop.pt
python pca_learn_dist.py -t resnet101_dolg --imsize 512 --embed_dim 128 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_dolg_pca_128d_gldv2_512x512_p3_global_randcrop.pt
python pca_learn_dist.py -t resnet101_dolg --imsize 512 --embed_dim 64 -b 32 -p 3 --num_samples -1 --dump_to ./pretrained_models/pca_weights/resnet101_dolg_pca_64d_gldv2_512x512_p3_global_randcrop.pt