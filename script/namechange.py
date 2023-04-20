# Change the names of all the files in a directory to be of form "000000.png", "000001.png", etc.

import os

# Set the directory you want to start from
rootDir = '/home/yoraish/classes/l3d/project/fish-nerf/tartanair/Sewerage/Data_hard/P000/image_lcam_fish'

# All the file names.
fnames = os.listdir(rootDir)

# Sort the file names.
fnames.sort()

# Iterate over the file names.
for i, fname in enumerate(fnames):
    # Get the old file name.
    old_fname = os.path.join(rootDir, fname)
    
    # Get the new file name.
    new_fname = os.path.join(rootDir, '{:06d}.png'.format(i))
    
    # Rename the file.
    print('Renaming {} to {}'.format(old_fname, new_fname))
    os.rename(old_fname, new_fname)