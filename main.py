from datasets.hapt_dataset import HaptDataset

train_x_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\X_train.txt"
train_y_data_path = r"C:\Code\Projects\Dance\Data\HAPT Data Set\Train\y_train.txt"

dataset = HaptDataset(train_x_data_path, train_y_data_path)