from .._utils import *
from logparser.Drain import LogParser
from ast import literal_eval
import datetime

class LogParse:
    def __init__(self):
        self.log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
        # Regular expression list for optional preprocessing (default: [])
        self.regex = [
            r"blk_(|-)[0-9]+",  # block id
            r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP
            r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
        ]
        self.st = 0.5  # Similarity threshold
        self.depth = 4  # Depth of all leaf nodes
    
    def set_format(self, format: str, regex):
        self.log_format = format
        self.regex = regex
        
        print(f"Updated log format: {self.log_format}")
        print(f"Updated regex patterns: {self.regex}")

    def get_csv(self, input_dir, output_dir, log_file):
        print("Start Parsing logs")
        parser = LogParser(log_format=self.log_format,indir=input_dir,outdir=output_dir,depth=self.depth,st=self.st,rex=self.regex)
        parser.parse(log_file)
        print("Finish Parsing")
        output_file = output_dir + log_file + "_structured.csv"
        return output_file

def get_BlockId(ParameterList):
    for s in ParameterList:
        sub = s.split(" ")
        # print(s)
        if "blk" in sub[0][:3]:
            return sub[0]
    return None

def get_datetime_from_timestamp(timestamp):
    df_datetime = timestamp.progress_apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
    df_datetime = df_datetime.progress_apply(lambda x: pd.to_datetime(str(x)))
    return df_datetime
    
def get_Datetime(date, time):
    df_date = date.progress_apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
    df_time = time.progress_apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S"))
    df_datetime = df_date + ' ' + df_time
    df_datetime = df_datetime.progress_apply(lambda x: pd.to_datetime(str(x)))
    return df_datetime

def get_hdfs_df(path: str, label_path: str):
    hdfs_df = pd.read_csv(path)
    
    print("Getting Pid process...")
    hdfs_df["Pid"] = hdfs_df["Pid"].astype(str)
    
    print("Getting DateTime process...")
    hdfs_df["DateTime"] = get_Datetime(hdfs_df["Date"], hdfs_df["Time"])
    
    print("literal_eval process...")
    hdfs_df["ParameterList"] = hdfs_df["ParameterList"].progress_apply(literal_eval)
 
    print("Getting BlockId process...")
    hdfs_df["BlockId"] = hdfs_df["ParameterList"].progress_apply(get_BlockId)
    hdfs_df = hdfs_df.dropna(subset="BlockId")
    
    print("Getting EventId process...")
    event_df = pd.DataFrame(([f"E{idx}", x] for idx, x in enumerate(hdfs_df["EventId"].unique())), columns=('event_id', 'EventId'))
    hdfs_df = hdfs_df.join(event_df.set_index("EventId"), on="EventId")
    hdfs_df = hdfs_df.reset_index(drop=True)

    label_df = pd.read_csv(label_path)
    print("Join the dataframe with labels")
    hdfs_df = hdfs_df.join(label_df.set_index("BlockId"), on="BlockId")

    hdfs_df = hdfs_df.reset_index(drop=True)

    print("Returning dataframe")
    return pd.DataFrame(
        {
            "date_time": hdfs_df["DateTime"],
            "component": hdfs_df["Component"],
            "session_id": hdfs_df["BlockId"],
            "pid": hdfs_df["Pid"],
            "severity": hdfs_df["Level"],
            "content": hdfs_df["Content"],
            "event_id": hdfs_df["event_id"],
            "event_template": hdfs_df["EventTemplate"],
            "label": hdfs_df["Label"],
        }
    )
    
def get_bgl_df(path: str):
    bgl_df = pd.read_csv(path)
    
    print("Getting DateTime process...")
    bgl_df["DateTime"] = get_Datetime(bgl_df["Date"], bgl_df["Timestamp"])
    
    bgl_df = bgl_df.reset_index(drop=True)

    print("Returning dataframe")
    return pd.DataFrame(
        {
            "date_time": bgl_df["DateTime"],
            "session_id": bgl_df["Node"],
            "severity": bgl_df["Level"],
            "content": bgl_df["Content"],
            "event_id": bgl_df["EventId"],
            "event_template": bgl_df["EventTemplate"],
            "label": bgl_df["Label"],
        }   
    )