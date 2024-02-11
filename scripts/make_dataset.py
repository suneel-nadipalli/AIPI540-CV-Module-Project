import opendatasets as od

def download_data():

    od.download("https://www.kaggle.com/datasets/jayswayambunathan/birds-and-squirrels/data", data_dir=".\\data\\raw", force=True)
