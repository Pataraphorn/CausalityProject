from .._utils import *

crashed = ['request from flp expert.',
           'ali-ecs core restarted',
           'mch-qcmn-epn-full-track-matching crashed.',
           'triggers out synch. disabling ctp readout.',
           'trigger readout went to zero.',
           'end calibration',
           'end test',
           'change luminosity',
           'tpc scan',
           'rm asks to stop the run',
           'sor call failed',
           'operator error: run restarted the same partition.',
           'need to restart trigger.',
           ]

def get_run_id(df: pd.DataFrame, run_df: pd.DataFrame):
    df['RunID'] = 0
    run_df.dropna(subset=['time_o2_start'], inplace=True)
    run_df.dropna(subset=['time_o2_end'], inplace=True)
    
    for idx in tqdm(range(len(run_df)), total=len(run_df)):
        start_date = run_df['time_o2_start'].iloc[idx]
        end_date = run_df['time_o2_end'].iloc[idx]
        
        if type(start_date) == float and type(end_date) == float:
            continue
        else : 
            df.loc[(df['date'] >= start_date) & (df['date'] <= end_date),'RunID'] = run_df['id'].iloc[idx]
            df.loc[(df['date'] >= start_date) & (df['date'] <= end_date),'RunQuality'] = run_df['run_quality'].iloc[idx]
    return df[df['RunID'] != 0]

def get_eor(df: pd.DataFrame, eor_df: pd.DataFrame):
    df['EOR'] = None
    eor_id = eor_df['run_id']
    
    for idx in tqdm(range(len(eor_id)), total=len(eor_id)):
        description = eor_df['description'].iloc[idx]
        reason_id = eor_df['reason_type_id'].iloc[idx]
        df.loc[(df['RunID'] == eor_id.iloc[idx]) ,'EOR'] = description
        df.loc[(df['RunID'] == eor_id.iloc[idx]) ,'EORTypeID'] = reason_id
    return df[df['EOR']!=None]

def normalize(text):
    if type(text) == str:
        text = text.lower()
        text = text.replace('of ', '')
        text = text.replace('in ', '')
        return text
    else :
        return text

def get_df(df: pd.DataFrame, label_path: str, eor_path: str, save_path: str = None):
    run_df = pd.read_csv(label_path)
    eor_df = pd.read_csv(eor_path)

    print("Getting DateTime process...")
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s')

    print('Mapping RunID from label file')
    df = get_run_id(df, run_df)

    print('Mapping EOR from EOR file')
    df = get_eor(df, eor_df)

    print('Removing all severity D')
    df = df[df['Severity'] != 'D']

    print('Normalization EOR')
    df['EOR'] = df['EOR'].progress_apply(normalize)
    df = df.dropna(subset='EOR')

    print('Getting Crashed from EOR')
    df['crash'] = ['crashed' if eor in crashed else 'not crashed' for eor in df['EOR'].to_list()]

    print("Getting EventId process...")
    template_df = pd.DataFrame(([f"E{idx}", x] for idx, x in enumerate(df["Template"].unique())), columns=('EventId', 'EventTemplate'))
    template_df.to_csv(save_path + '/log_templates.csv')  

    df = df.merge(template_df, left_on='Template', right_on='EventTemplate', how='left')

    df = df.reset_index(drop=True)
    print(df.head())

    print("Returning dataframe")
    return pd.DataFrame(
        {
            "date_time": df["date"],
            "session_id": df["RunID"],
            "severity": df["Severity"],
            "content": df["Content"],
            "event_id": df["PID"],
            "event_id": df["EventId"],
            "event_template": df["Template"],
            "label": df["crash"],
            "EOR": df["EOR"],
            "EOR_id": df["EORTypeID"],
            "run_quality": df["RunQuality"],
        }
    )
