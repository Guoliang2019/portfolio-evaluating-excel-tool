import pandas as pd

def index_matching(idx1,idx2):
    '''
    Generating mapping table for similar string so as to achieve consistent naming
        1. 'XX_XX' <--> 'XX XX'  
        2. 'xxYYY' <--> 'xx'
    Parameters:
    --------- 
    idx1: array-like whose elements are string
    idx2: array-like whose elements are string
    
    Return:
    ------
    mapping,dict
    
    '''
    mapping = {}
    for i1 in idx1:
        for i2 in idx2:
            underscore1 = i1.replace('_',' ')
            underscore2 = i2.replace('_',' ')
            if underscore1 == underscore2:
                mapping[i1]=underscore1
                mapping[i2]=underscore2
            else:
                cap1 = i1.upper()
                cap2 = i2.upper()
                if cap1 in i2:
                    mapping[i1] = i2
                    mapping[i2] = i2
                elif cap2 in i1:
                    mapping[i2] = i1
                    mapping[i1] = i1
                else:
                    continue
    return mapping   


_label_of_setting = ['Portfolio Name','Portfolio Frequency','Rolling Window Size','Momentum N','Cash Excluded']

def label_matching(l,label_of_setting=_label_of_setting):
    '''
    Generate mapping dict for string elements in two equal-length list_like object
    
    Parameters:
    -----------
    l: list_like, elements of which are string
    label_of_setting: list, elements of which are string
    
    Returns:
    --------
    mapping: dict
    
    '''
    mapping = {}
    for i1 in l:
        for i2 in label_of_setting:
            if i2 in i1:
                mapping[i1]=i2
    return mapping

def parse_settable_input(inputs):
    '''
    Parse Excel spreed sheet for settable inputs and divide into two dataframes for portfolio setting and benchmark setting
    
    Parameters:
    -----------
    inputs: pandas dataframe, no header and no index with first columns as reference to separate portfolio's information and benchmark's
    
    Returns:
    --------
    portfolio_set: pandas dataframe, indexed by portfolio settings' name, consistent with the default _label_of_setting
    benchmark_set: pandas dataframe, indexed by market indices and headered by customized benchmark name
    
    '''
    inputs = inputs.iloc[:20,:12].dropna(axis=1,how='all').dropna(axis=0,how='all')
    inputs.index = range(inputs.shape[0])
    
    iden = inputs.index[inputs[0]=='Benchmark Constituents'][0] # use first columns NA to identify portfolio and benchmark settings
    
    portfolio_set = inputs.iloc[:iden,:].dropna(axis=1,how='all')
    benchmark_set = inputs.iloc[iden:,:].dropna(axis=1,how='all')
    
    benchmark_set.set_index(benchmark_set.columns[0],inplace=True)
    benchmark_set = benchmark_set.T
    benchmark_set.set_index(benchmark_set.columns[0],inplace=True)
    benchmark_set = benchmark_set.T
    benchmark_set.index.name = None
    benchmark_set = benchmark_set.astype(float)
    
    portfolio_set.set_index(portfolio_set.columns[0],inplace=True)
    portfolio_set.rename(index=label_matching(portfolio_set.index),inplace=True)
    portfolio_set.index.name = None
    
    portfolio_set = parse_portfolio_input(portfolio_set)
    
    return portfolio_set,benchmark_set

def parse_portfolio_input(portfolio_set):
    label_of_setting=list(portfolio_set.index)
    
    def one_element_list(lst):
        if len(lst) == 1:
            return lst[0]
        else:
            return lst
    
    return dict(zip(label_of_setting, [one_element_list(list(portfolio_set.loc[label].dropna()))\
                                       for label in label_of_setting]))

def parse_benchmark_input(benchmark_set,date_index):
    portfolios = benchmark_set.columns
    assets = benchmark_set.index
    one_day = pd.concat([benchmark_set[k] for k in portfolios],axis=0)
    one_day.index = pd.MultiIndex.from_product([portfolios,assets],names=['Portfolio','Asset'])
    df = pd.concat([one_day]*len(date_index),axis=1).T
    df.index = date_index
    return df