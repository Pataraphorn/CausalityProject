from .._utils import *
from . import alice_info, logParse

class preprocess:
    def __init__(self, data_type="hdfs", parsing=True):
        super().__init__()
        self.data_type = data_type
        self.parsing = parsing
        self.csv_path = None
        self.df = pd.DataFrame()
        print("Creating Preprocessing Data tpye", self.data_type)

    def log_parsing(self, input_dir, output_dir, log_file):
        if self.data_type == "hdfs":
            self.csv_path = logParse.LogParse().get_csv(
                input_dir=input_dir, output_dir=output_dir, log_file=log_file
            )
        elif self.data_type == "bgl":
            logParse.LogParse().set_format(format = "<Label> <Time> <Date> <Node> <Timestamp> <NodeRepeat> <Type> <Component> <Level> <Content>", regex = [r"(?P<Label>[A-Z]+|-)", r"core\.\d+", r"(?P<Content>.+)"])
            self.csv_path = logParse.LogParse().get_csv(
                input_dir=input_dir, output_dir=output_dir, log_file=log_file
            )
        else:
            print(f"This version not having log_parsing for {self.data_type} type yet.")
            pass
        return self.csv_path

    def from_parsed_csv(self, csv_path):
        self.csv_path = csv_path
        return self.csv_path

    def load_data_from_path(self, path: str, type: str = None):
        self.df = load_df_from_path(path, type)
        return self.df

    def prep_data(self, label_path: str = None, eor_path: str = None, save_path: str = None):
        print(f"Preprocessing with label from {self.data_type}")
        print(f"Path :{label_path}")
        if self.data_type == "hdfs":
            self.df = logParse.get_hdfs_df(self.csv_path, label_path)
        if self.data_type == "bgl":
            self.df = logParse.get_bgl_df(self.csv_path)
        elif self.data_type == "alice_info":
            self.df = alice_info.get_df(self.df, label_path, eor_path, save_path)
        else:
            print(f"This version not having prep_data for {self.data_type} type yet.")
            pass
        print("Done")

    def set_df(self, df):
        self.df = df

    def save_flie(self, save_path: str):
        self.df.to_parquet(save_path,compression='gzip')  

def show_details_df(df: str):
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Info of data: \n{df.info()}")
