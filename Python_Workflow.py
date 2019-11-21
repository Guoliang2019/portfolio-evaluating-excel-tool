from Parsing_Input import *
from additional_help import *
from Computing_Single_Strategy import *

def clean_up(settable_inputs,fund_weights,custom_returns,equity_returns,benchmark_returns):
    portfolio_set,benchmark_set = parse_settable_input(settable_inputs)
    
    
    
    Portfolio_Name = portfolio_set['Portfolio Name']
    Portfolio_Frequency = int(portfolio_set['Portfolio Frequency'])
    Rolling_Window_Size = int(portfolio_set['Rolling Window Size'])
    Momentum_N =  int(portfolio_set['Momentum N'])
    Cash_Excluded = portfolio_set['Cash Excluded']
    
    ### parse strategy-related input
    fund_weights.sort_index(inplace=True)
    equity_returns.sort_index(inplace=True)
    custom_returns.sort_index(inplace=True)
    fund_weights.rename_axis(mapper=['Portfolio','Asset'],axis=1,inplace=True) # Give headers layer name

    if Cash_Excluded:
        if 'cash' in fund_weights.columns.get_level_values('Asset'):
            fund_weights.drop(columns=['cash'],level='Asset',inplace=True) # Drop cash if needed by default remove cash asset


    equity_returns = equity_returns/100 # raw data is in percentage 
    fund_returns = custom_returns.join(equity_returns,how='outer').loc[fund_weights.index.min():fund_weights.index.max()] # combine ret
    fund_rename_mapping = index_matching(fund_weights.columns.get_level_values('Asset').unique(),fund_returns.columns)
    fund_weights.rename(columns=fund_rename_mapping,level='Asset',inplace=True)
    fund_returns.rename(columns=fund_rename_mapping,inplace=True)
    
    
    benchmark_returns.sort_index(inplace=True)
    benchmark_returns = benchmark_returns/100
    
    benchmark_weights=parse_benchmark_input(benchmark_set,fund_weights.index.copy())
    benchmark_rename_mapping = index_matching(benchmark_weights.columns.get_level_values('Asset').unique(),benchmark_returns.columns)
    benchmark_weights.rename(columns=benchmark_rename_mapping,level='Asset',inplace=True)
    benchmark_returns.rename(columns=benchmark_rename_mapping,inplace=True)
    
    cleaned_data = {'Portfolio_Name':Portfolio_Name,'Portfolio_Frequency':Portfolio_Frequency,'Rolling_Window_Size':Rolling_Window_Size,\
                    'Momentum_N':Momentum_N,'Cash_Excuded':Cash_Excluded,'fund_weights':fund_weights,'fund_returns':fund_returns,\
                    'benchmark_returns':benchmark_returns,'benchmark_weights':benchmark_weights}
    return cleaned_data
    

def main(settable_inputs,fund_weights,custom_returns,equity_returns,benchmark_returns):
        
    portfolio_set,benchmark_set = parse_settable_input(settable_inputs)
    
    
    
    Portfolio_Name = portfolio_set['Portfolio Name']
    Portfolio_Frequency = int(portfolio_set['Portfolio Frequency'])
    Rolling_Window_Size = int(portfolio_set['Rolling Window Size'])
    Momentum_N =  int(portfolio_set['Momentum N'])
    Cash_Excluded = portfolio_set['Cash Excluded']
    
    ### parse strategy-related input
    fund_weights.sort_index(inplace=True)
    equity_returns.sort_index(inplace=True)
    custom_returns.sort_index(inplace=True)
    fund_weights.rename_axis(mapper=['Portfolio','Asset'],axis=1,inplace=True) # Give headers layer name

    if Cash_Excluded:
        if 'cash' in fund_weights.columns.get_level_values('Asset'):
            fund_weights.drop(columns=['cash'],level='Asset',inplace=True) # Drop cash if needed by default remove cash asset


    equity_returns = equity_returns/100 # raw data is in percentage 
    fund_returns = custom_returns.join(equity_returns,how='outer').loc[fund_weights.index.min():fund_weights.index.max()] # combine ret
    fund_rename_mapping = index_matching(fund_weights.columns.get_level_values('Asset').unique(),fund_returns.columns)
    fund_weights.rename(columns=fund_rename_mapping,level='Asset',inplace=True)
    fund_returns.rename(columns=fund_rename_mapping,inplace=True)
    
    
    benchmark_returns.sort_index(inplace=True)
    benchmark_returns = benchmark_returns/100
    
    benchmark_weights=parse_benchmark_input(benchmark_set,fund_weights.index.copy())
    benchmark_rename_mapping = index_matching(benchmark_weights.columns.get_level_values('Asset').unique(),benchmark_returns.columns)
    benchmark_weights.rename(columns=benchmark_rename_mapping,level='Asset',inplace=True)
    benchmark_returns.rename(columns=benchmark_rename_mapping,inplace=True)
    
    fund_portfolio_returns = fund_weights.groupby(axis=1,level='Portfolio').\
    apply(lambda f: calculate_portfolio_return(f[f.name],fund_returns,f.name))
    
    benchmark_portfolio_returns = benchmark_weights.groupby(axis=1,level='Portfolio').\
    apply(lambda f: calculate_portfolio_return(f[f.name],benchmark_returns,f.name))
    
    eq_portfolio_weights = make_equal_weight_copy(fund_weights)
    
    eq_portfolio_returns = eq_portfolio_weights.groupby(axis=1,level='Portfolio').\
    apply(lambda f: calculate_portfolio_return(f[f.name],fund_returns,f.name))
    
    bag_of_portfolio_returns = fund_portfolio_returns.join([eq_portfolio_returns,benchmark_portfolio_returns])
    
    common_statistics = bag_of_portfolio_returns.apply(lambda f: calculate_common_statistics(f,Portfolio_Frequency,Momentum_N),axis=0)
    
    benchmark_statistics = pd.concat(\
    [fund_portfolio_returns.apply(lambda f: calculate_scalar_statistics(f,Portfolio_Frequency,Momentum_N,benchmark_portfolio_returns[b]),\
                                     axis=0) for b in benchmark_portfolio_returns.columns],axis=0)
    
    benchmark_statistics.index = pd.MultiIndex.\
    from_tuples([tuple(ind.split(' vs ')) for ind in benchmark_statistics.index],names=['Statistic','Benchmark'])
    benchmark_statistics = benchmark_statistics.unstack().stack(level='Portfolio')
    benchmark_statistics.index = [i[0]+' '+i[1] for i in list(benchmark_statistics.index)]
    
    Portfolio_Statistics =\
    pd.concat([common_statistics.reset_index(),benchmark_statistics.reset_index()],axis=0,ignore_index=True,sort=False)
    
    Portfolio_Statistics.set_index('index',inplace=True)
    Portfolio_Statistics.index.name = 'Portfolio Statistics'
    
    constant_param_mapper = {'freq':Portfolio_Frequency,'month_size':Rolling_Window_Size,'N':Momentum_N,'rf':0}
    func_mapper = get_portfolio_return_statistics_func_mapper()
    
    func_bag_of_portfolio_returns = ['Latest Annualized Return','Latest Annualized Standard Deviation','Drawdown']
    func_fund_eq_returns = ['Calendar Year Return','Month Rolling Return']
    
    fund_eq_portfolio_returns = fund_portfolio_returns.join(eq_portfolio_returns)
    
    result_bag_of_portfolio_returns = dict([(func,bag_of_portfolio_returns.apply(lambda f: \
    func_mapper[func](**get_portfolio_return_statistics_param_mapper(portfolio_return=f,**constant_param_mapper)[func]),\
                                                                                 axis=0)) for func in func_bag_of_portfolio_returns])
    result_fund_eq_portfolio = dict([(func,fund_eq_portfolio_returns.apply(lambda f: \
    func_mapper[func](**get_portfolio_return_statistics_param_mapper(portfolio_return=f,**constant_param_mapper)[func]),\
                                                                                 axis=0)) for func in func_fund_eq_returns])
    
    correlation = fund_eq_portfolio_returns.corr()
    correlation.index.name = 'Correlation'
    
    result = {'Portfolio Statistics':Portfolio_Statistics,'Correlation':correlation,'Return':bag_of_portfolio_returns}
    result.update(result_bag_of_portfolio_returns)
    result.update(result_fund_eq_portfolio)
    
    return result