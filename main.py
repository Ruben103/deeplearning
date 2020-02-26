import data as dt
import model
import visualasiation as vis

toggleaugmentation = 1
toggle_large_data = 1

if __name__ == '__main__':

    dat = dt.Data()
    dt.face_extr()
    train_data, train_labels, test_data, test_labels, width, height = dt.load_data_test()

    mod = model.LeNet()
    mod.model_test(train_data, train_labels, test_data, test_labels, toggleaugmentation, width, height)

    # data = dt.load_full_data()
    print(dat.shape)
    vis.multi(dat)

    dingutje = model.LeNet()

    print("BUGSTOPP")