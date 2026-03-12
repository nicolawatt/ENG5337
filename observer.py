#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#------------------------------Lab 3 - Observer-------------------------------#
#-----------------------------------------------------------------------------#

# IMPORTANT: display names must match the Probe names in cnn_pid_line_follower.py
# and pid_baseline.py exactly — mismatch causes "Unable to receive data" error

# Launch order: run the main script FIRST, wait for "Ready" message,
# THEN run this observer script in a second terminal.

from pal.utilities.probe import Observer
import time

time.sleep(2)  # wait for Probe to initialise before connecting

observer = Observer()
observer.add_display(imageSize=[200, 320, 1],
                    scalingFactor=2,
                    name='Raw Image')
observer.add_display(imageSize=[50, 320, 1],
                    scalingFactor=1,
                    name='Binary Image')
observer.launch()