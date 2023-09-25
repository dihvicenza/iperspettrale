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
# When created, the DataReader instance loads a hyperspectral dataset and performs calibration.
# The user can then visualize the results for further analysis. 
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
        if target in os.listdir("./calibrated"): # if calibration files already exist, simply load them from the "calibrated" folder; otherwise, generate them
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

    # Get the heatmap of the image at a specific bandwidth.
    # The selected band will be the closest one among the band buckets included in the source hdr file.
    def get_heatmap(self, target_band):
        selected_band = self.calibrated_np[:, :, self.get_band_index(target_band)]
        return selected_band

    # Display the heatmap of the image at a specific bandwidth.
    def show_heatmap(self, target_band, save_image=False):
        selected_band = self.get_heatmap(target_band)

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
    
    # Show the reflectance across all bands for a given pixel.
    # OUTPUT: 
    #   1D list of band values.
    #   1D np array of reflectance values.
    def show_pixel_response(self, x, y):
        pixel_reflectance = self.calibrated_np[y:y+1, x:x+1,:]
        pixel_squeezed = np.squeeze(pixel_reflectance)
        plt.plot(self.bands, pixel_squeezed)
        plt.title('Spectral Footprint\n(Pixel {},{})'.format(x, y))
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.show()
        return self.bands, pixel_squeezed

    # Get the reflectance averaged over a given pixel region.
    # INPUT: 
    #   x, y - top left corner of region of interest.
    #   w, h - width and height of region of interest, with corner at (x, y).
    # OUTPUT: 
    #   1D list of band values.
    #   1D np array of reflectance values.
    def get_region_response(self, x, y, w, h):
        region_reflectance = self.calibrated_np[y:y+h, x:x+w,:]
        reflectance = []
        for band_index in range(region_reflectance.shape[2]):
            reflectance.append(np.mean(region_reflectance[:,:,band_index]))
        return self.bands, reflectance

    # Display the reflectance averaged over a given pixel region.
    # INPUT: 
    #   x, y - top left corner of region of interest.
    #   w, h - width and height of region of interest, with corner at (x, y).
    #   wavelengths - array of bands to mark
    # OUTPUT: 
    #   1D list of band values.
    #   1D np array of reflectance values.
    def show_region_response(self, x, y, w, h, wavelengths=[]):
        _, reflectance = self.get_region_response(x, y, w, h)
        plt.plot(self.bands, reflectance)
        plt.title('Spectral Footprint\n(Region {},{})'.format(x, y))
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.vlines(x=[self.get_band_index(w) for w in wavelengths], ymin=min(reflectance), ymax=max(reflectance), colors='red', ls='--', lw=2)
        plt.show()

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
    dr.show_region_response(600, 100, 5, 5, [1650])
    # dr.show_heatmap(1650, save_image=True)