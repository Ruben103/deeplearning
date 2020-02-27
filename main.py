import data as dt
import model
import visualasiation as vis
import os

toggleaug = 0
toggle_large_data = 1

if __name__ == '__main__':

    dat = dt.Data()
    data_full = dat.load_full_data()
    vis.multi(data_full)

    train_data, train_labels, width, height = dat.load_data_test(toggle_large_data)

    mod = model.LeNet()
    mod.model_test(train_data, train_labels,toggleaug, width, height)

    # data = dt.load_full_data()
    #print(dat.shape)
    #vis.multi(dat)

    #dingutje = model.LeNet()

    print("BUGSTOPP")