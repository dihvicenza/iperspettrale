# ------------------------------------------------------------------------------------------------------
# DIGITAL INNOVATION HUB VICENZA
# Author: Gabriele Sha

# Sources
#   https://tobinghd.wordpress.com/2020/10/28/reading-a-hyperspectral-image-python/
#   https://eufat.github.io/2019/02/19/hyperspectral-image-preprocessing-with-python.html
#   https://www.spectralpython.net/graphics.html

# This code includes functionality for reading and visualizing hyperspectral data in HDR format. 
# Note: Datasets must include .hdr, .raw and .log files.
# ------------------------------------------------------------------------------------------------------

import wx
import spectral as sp
from spectral import imshow, view_cube
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

print("Spectral version:", sp.__version__)

# ------------------------------------------------------------------------------------------------------
# DATAREADER
# ------------------------------------------------------------------------------------------------------
# The DataReader class is an object that loads a hyperspectral dataset, performs calibration, and allows
# the user to visualize the results for further analysis. 
# 
# INPUTS
#   - dataset_path: path to a directory containing all relevant datasets (default current directory)
#   - hdr_path: must point to the "capture" folder containing DARKREF, WHITEREF, and raw data .hdr files

class DataReader:

    def __init__(self, dataset_path= "./", hdr_path="HYPERSPECTRAL DATSASET2_14-09-2023/jeans_Elastico_chiaro_2023-09-12_07-32-56/capture/"):
        data_path = os.path.join(dataset_path, hdr_path)
        sp.settings.envi_support_nonlowercase_params = True 
        dir_list = os.listdir(data_path)
        
        for file in dir_list:
            if file.endswith(".hdr"):
                if file.startswith("WHITEREF"):
                    print(data_path + file)
                    self.white_ref = envi.open(data_path + file)
                elif file.startswith("DARKREF"):
                    print(data_path + file)
                    self.dark_ref = envi.open(data_path + file)
                else:
                    print(data_path + file)
                    self.hdr_path = data_path + file
                    self.data = envi.open(self.hdr_path)
        
        self.bands = sp.envi.read_envi_header(self.hdr_path)['Wavelength']

        target = self.hdr_path.split('/')[-1][:-4] + '_calibrated.npy'
        if target in os.listdir("./calibrated"):
            self.calibrated_np = np.load("./calibrated/" + target)
        else:
            self.calibrate()
    
    # Print the header list, containing information about the dataset.
    def print_header(self): 
        h = sp.envi.read_envi_header(self.hdr_path)
        print(h)

    # Print the list of bands over which the dataset was acquired. 
    def print_bands(self):
        print(self.bands)

    # Print basic information about the dataset: Size and number of bands.
    def print_description(self):
        wvl = self.data.bands.centers
        rows, cols, bands = self.data.nrows, self.data.ncols, self.data.nbands
        meta = self.data.metadata

        print(meta)
        print(f"Wavelengths: {wvl}")
        print(f"Num rows: {rows}\nNum cols: {cols}\nNum bands: {bands}")
    
    # Calibrate the dataset using white and dark reference hdr files. These must all be contained in the same
    # "capture" directory. The resulting numpy data structure is stored in the "./calibrated" folder as a numpy
    # array, as well as within the DataReader instance as self.calibrated_np.
    def calibrate(self):
        print("CALIBRATING...")
        white_np = np.array(self.white_ref.load())
        dark_np = np.array(self.dark_ref.load())
        data_np = np.array(self.data.load())

        self.calibrated_np = np.empty((0,self.data.ncols,self.data.nbands))

        x_step = white_np.shape[0]

        for i in range(0, len(data_np), x_step):
            if i + x_step > len(data_np):
                break
            group = data_np[i:i+x_step]
            group = np.divide(np.subtract(group, dark_np), np.subtract(white_np, dark_np))
            self.calibrated_np = np.concatenate((self.calibrated_np, group), axis=0)
            print(self.calibrated_np.shape)
        
        print("CALIBRATION DONE")

        np.save("./calibrated/" + self.hdr_path.split('/')[-1][:-4] + '_calibrated',self.calibrated_np)

        print("Saved calibrated image to ./calibrated/")

    # Show the full calibrated dataset as a hyperspectral cube.
    def show_calibrated(self):
        app = wx.App(False)  # Create a wx.App object
        img = self.calibrated_np        
        view_cube(img)
        app.MainLoop()

    # Show the reflectance across all bands for a given pixel.
    def show_pixel_response(self, leaf_pixel_x, leaf_pixel_y):
        leaf_pixel = self.calibrated_np[leaf_pixel_y:leaf_pixel_y+1, leaf_pixel_x:leaf_pixel_x+1,:]
        leaf_pixel_squeezed = np.squeeze(leaf_pixel)
        plt.plot(self.bands, leaf_pixel_squeezed)
        plt.title('Leaf Spectral Footprint\n(Pixel {},{})'.format(
            leaf_pixel_x, leaf_pixel_y))
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.show()

    # Displays the heatmap of the image at a specific bandwidth (the selected band will be the closest one
    # among the band buckets included in the source hdr file).
    def create_heatmap(self, target_band, save_image=False):
        selected_band = self.calibrated_np[:, :, self.get_band_index(target_band)]

        if save_image:
            img_name = "images/" + self.hdr_path.split('/')[-1][:-4] + ".jpg"
            normalized = selected_band / np.max(selected_band)
            colormap = plt.cm.jet
            colored_data = (colormap(normalized) * 255).astype(np.uint8)
            img = Image.fromarray(colored_data)
            rgb_im = img.convert('RGB') # remove alpha channel for JPG
            rgb_im.save(img_name)

        fig, ax = plt.subplots(figsize=(4, 4))  # Adjust the figsize as needed
        plt.imshow(selected_band, cmap='jet')  # Use an appropriate colormap
        plt.colorbar(label='Reflectance')  # Add a colorbar with a label
        plt.suptitle(f'Reflectance Heatmap ({target_band}nm)',fontsize=12)
        plt.title(self.hdr_path.split('/')[-1],fontsize=8)
        plt.xlabel('Pixel Column')  # Label for the x-axis
        plt.ylabel('Pixel Row')  # Label for the y-axis
        ax.set_position([0.3, .1, .2, .8])  # [left, bottom, width, height]
        plt.show()

    def get_band_index(self, target_band):
        i = 0
        while float(self.bands[i]) < target_band:
            i += 1
        return i   

# ------------------------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------------------------

if __name__=="__main__":
    
    dr = DataReader(dataset_path="../datasets/", hdr_path="HYPERSPECTRAL DATSASET2_14-09-2023/jeans_Elastico_chiaro_2023-09-12_07-32-56/capture/")
    # dr = DataReader(dataset_path="../datasets/", hdr_path="HYPERSPECTRAL DATSASET2_14-09-2023/jeans_Elastico_scuro_2023-09-12_07-31-45/capture/")
    # dr = DataReader(dataset_path="../datasets/", hdr_path="HYPERSPECTRAL DATSASET2_14-09-2023/jeans_NoElastico_chiaro_2023-09-12_07-34-12/capture/")
    
    # dr.print_header()
    # dr.print_bands()
    # dr.show_calibrated()
    # dr.show_pixel_response(600, 100)
    dr.create_heatmap(1650, save_image=True)