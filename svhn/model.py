from svhn_data import load_svhn_data
import PIL.Image as Image
import matplotlib.pyplot as plt



test_data, test_labels = load_svhn_data("test", "full")

print(test_data.shape)
print(test_labels.shape)

