import numpy as np
import matplotlib.pyplot as plt


num_exec_list = np.loadtxt('/home/wiser-renjie/projects/blockcopy/Pedestron/results/csp_blockcopy_050_wildtrack_c1_c7_num_exec_list.txt')

# Step 2: Create histogram
plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.hist(num_exec_list, bins=10, color='blue', alpha=0.7, edgecolor='black')

# Labeling, titles and limits
plt.xlabel('Percentage of executed blocks')
plt.ylabel('# Images')
plt.title('Distribution of Executed Blocks')

# Show plot
plt.grid(True)  # Show grid (optional)
plt.tight_layout()  # Fit the plot nicely to the figure (optional)

# Save plot as an image file
plt.savefig('histogram.png')  # Save as a .png file
plt.close()  # Close the figure