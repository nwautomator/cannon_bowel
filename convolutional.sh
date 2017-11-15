#!/usr/bin/bash

python ./mrc_3d_bysection.py 28May_original.mrc x0.mrc
tar -cvf pass0.jpgs.tar wo*jpg
python ./mrc_3d_bysection.py x0.mrc x1.mrc
tar -cvf pass1.jpgs.tar wo*jpg
python ./mrc_3d_bysection.py x1.mrc x2.mrc
tar -cvf pass2.jpgs.tar wo*jpg
python ./mrc_3d_bysection.py x2.mrc x3.mrc
