import data as dt
import model
import os
import visualisation as vis

toggleaug = 0
toggle_large_data = 0

architect = 0 # 0 == architecture 2

if __name__ == '__main__':

    dat = dt.Data()
    # dat.face_extr()

    # toggle comment to visualize the data
    data_full = dat.load_full_data()
    vis.multi(data_full)

    # function to return the data
    train_data, train_labels, test_data, test_labels, width, height = dat.load_data_test(toggle_large_data)

    # create model and run test
    mod = model.LeNet()
    mod.model_test(train_data, train_labels, test_data, test_labels, toggleaug, width, height, architect)
