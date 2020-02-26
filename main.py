import data as dt
import model
import visualasiation as vis

toggleaugmentation = 1
toggle_large_data = 0

if __name__ == '__main__':

    dat = dt.Data()
    #dat.face_extr()
    train_data, train_labels, width, height = dat.load_data_test(toggle_large_data)

    mod = model.LeNet()
    mod.model_test(train_data, train_labels)

    # data = dt.load_full_data()
    #print(dat.shape)
    #vis.multi(dat)

    #dingutje = model.LeNet()

    print("BUGSTOPP")