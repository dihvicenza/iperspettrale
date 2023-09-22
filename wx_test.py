# This script is only used for testing wxPython GUI functionality

import wx
import matplotlib.pyplot as plt

app = wx.App(False)
frame = wx.Frame(None, wx.ID_ANY, "Hello wxPython")
frame.Show(True)
app.MainLoop()


def create_window(self): # DEBUG: wx test
    app = wx.App(False)  
    frame = wx.Frame(None, wx.ID_ANY, "Hyperspectral Viewer")
    frame.Show(True)
    # plt.imshow(img, cmap='jet')
    plt.colorbar()
    plt.title('Hyperspectral Band')
    plt.show()
    app.MainLoop()
