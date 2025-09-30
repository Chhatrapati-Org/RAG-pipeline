import os
from natsort import natsorted

f = open(r"json_file\out.json", "wb")
files = os.listdir(r"C:\Users\22bcscs055\Downloads\mock_data")
files = natsorted(files)


for fname in files:
    print(rf"C:\Users\22bcscs055\Downloads\mock_data\{fname}")
    f.write(open(rf"C:\Users\22bcscs055\Downloads\mock_data\{fname}", "rb").read())
    f.flush()

f.close()