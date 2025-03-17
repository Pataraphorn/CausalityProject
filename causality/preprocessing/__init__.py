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
        print(f"Load data from type:{type} in path:{path} to Dataframe")
        if type == "parquet":
            self.df = pd.read_parquet(path)
        elif type == "npz":
            npz = np.load(path, allow_pickle=True)
            self.df = pd.DataFrame({file: npz[file] for file in npz.files})
        elif type == "pickle":
            self.df = pd.read_pickle(path)
        else:
            self.df = pd.read_csv(path)
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

    def topic_to_topic_id(self):
        print("Getting TopicId process...")
        topic_df = pd.DataFrame(
            (
                [f"{str(x).split('_')[0]}", x]
                for idx, x in enumerate(self.df["topic"].unique())
            ),
            columns=("topic_id", "topic"),
        )
        topic_df["topic_id"] = pd.to_numeric(topic_df["topic_id"], errors="coerce")
        topic_df["topic_id"] = topic_df["topic_id"].apply(
            lambda x: int(x) if pd.notna(x) and x.is_integer() else x
        )
        self.df = self.df.join(topic_df.set_index("topic"), on="topic")
        self.df = self.df.reset_index(drop=True)
        return self.df


def show_details_df(df: str):
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Info of data: \n{df.info()}")
