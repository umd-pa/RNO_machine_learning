from album_utils import copy_with_progress, trainTest_split
import sys
import os


album_dir = '/data/i3store/users/ssued/albums/RNO_album_12_30_2025_50k_cartesian'
album_name = 'album_RNO4.hdf5'

# Backup Album
#copy_with_progress(album_dir + '/' + album_name , album_dir + '/backup_' + album_name)

# Train Test Split

trainTest_split(album_dir,album_name,0.8,backup=False)