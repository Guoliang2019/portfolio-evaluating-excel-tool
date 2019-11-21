import pandas as pd

def make_equal_weight_copy(original):
    eq = original.copy()
    eq = eq.groupby(axis=1,level='Portfolio').apply(lambda f: pd.DataFrame(1/f.shape[1],columns=f.columns,index=f.index))
    eq_port_name = eq.columns.get_level_values('Portfolio')
    eq_port_renaming = dict(zip(eq_port_name,[p+' Equal Weight' for p in eq_port_name]))
    eq.rename(columns=eq_port_renaming,inplace=True,level='Portfolio')
    return eq

def process_row_col(df,row,col):
    df = df.copy()
    if row is not None:
        df = df.loc[row]
    if col is not None:
        df = df[col]
    return df

def find_strategy_benchmark(data,strategy_name,benchmark_name):
    common_idx = data.dropna(axis=0,how='any').index.copy()
    benchmark_idx = (data.index.difference(common_idx)).copy()
    benchmark_idx = benchmark_idx[benchmark_idx.str.find(strategy_name) != -1]
    idx = common_idx.append(benchmark_idx)
    return data.loc[idx,[strategy_name,benchmark_name]]