import pandas as pd
import numpy as np
import scipy.stats as st


def calculate_portfolio_return(weights,returns,name):
    '''
    Calculate portfolio return over its history based on weights and returns of assets
    
    portfolio_return(t) = weights(t-1) @ returns(t)
    
    Parameters
    -----------------
    weights: dataframe, time-series weights of a few assets
    returns: dataframe, time-series returns of a few assets 
    name: str, name of the strategy(return)
    
    Return
    -----------------
    Series: time-series returns with name
    '''
    uniqueness = weights.columns.is_unique
    matching = set(weights.columns).issubset(set(returns.columns))
    
    if uniqueness and matching: 
        returns = returns[weights.columns]
        output = pd.Series(index=weights.index.copy(),name=name)
        for t in output.index:
            #print(t)
            if t == output.index[0]: # first date of the portfolio has no return
                output[t] = 0
                pre_t = t
            else: # weights(t-1) @ returns(t) = portfolio_return(t)
                realized_ret = returns.loc[(returns.index > pre_t)&(returns.index<=t)].add(1).prod(min_count=1)-1
                output[t] = weights.loc[pre_t,:].dot(realized_ret)
                pre_t = t
        return output
    else:
        raise AttributeError('assets under weights are {} unique'.format('' if uniqueness else 'not')+'\n'+\
              '               assets under weights can {} be found under returns'.format('' if matching else 'not'))

def calculate_annualized_return(portfolio_return,freq,method='direct'):
    '''
    Calculate annualized return of a given portfolio return
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return of a strategy
    freq: int,frequency of the portfolio return(number of periods within a year)
    method: str, 'direct', total return --> annualized return
                 ∏(1+portfolio_return(t))^(freq/N) - 1, for all t
    
                 'estimate', average return --> annualized return
                 (1 + mean(portfolio_return(t)))^freq - 1, for all t
    Return
    -----------------
    result, float
    '''
    if method in ('direct','d'):
        N = len(portfolio_return)
        result = np.prod(portfolio_return+1)**(freq/N) - 1 
    elif method in ('estimate','e'):
        result = (1+np.mean(portfolio_return))**freq - 1
    else:
        raise NameError('Only accept method direct/d or estimate/e ')
    return result

def calculate_latest_annualized_returns(portfolio_return,freq):
    '''
    Calculate the latest 1/2/3/5/7/10 years annualized return.
    
    ∏(1+portfolio_return(t))^(freq/N) - 1, for all t starting from 1/2/3/5/7/10 years ago.
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    name: str, name of the corresponding strategy/portfolio return
    freq: int, frequency of portfolio(number of periods within a year)
    
    Returns
    -----------------
    series, annualized returns with index of name of length of history 
    '''
    lengths = [freq*i for i in [1,2,3,5,7,10] if freq*i <= len(portfolio_return)] #the maximum years allowed go back given portfolio history
    ann_rets = [calculate_annualized_return(portfolio_return.tail(l),freq=freq) for l in lengths]
    index = ["Latest Annualized Return {} Month".format(int(l/freq)*12) for l in lengths] # in terms of month

    return pd.Series(ann_rets,index=index,name=portfolio_return.name)

def calculate_annualized_stdev(portfolio_return,freq,ddof=1):
    '''
    Calculate annualized standard deviation of a given portfolio return
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return of a strategy
    freq: int,frequency of the portfolio return(number of periods within a year)
    
    Return
    -----------------
    result, float
    '''
    return np.std(portfolio_return,ddof=ddof)*np.sqrt(freq)

def calculate_latest_annualized_stdev(portfolio_return,freq):
    '''
    Calculate the latest 1/2/3/5/7/10 years annualized standard deviation.
    
    std(any_portfolio(t))*freq^0.5, for all t starting from 1/2/3/5/7/10 years ago.
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    name: str, name of the corresponding strategy
    freq: int, frequency of portfolio
    
    Returns
    -----------------
    series, annualized standard deviations with index of name of length of history 
    '''
    lengths = [freq*i for i in [1,2,3,5,7,10] if freq*i <= len(portfolio_return)] # the maximum years allowed go back given portfolio history
    ann_stds = [calculate_annualized_stdev(portfolio_return.tail(l),freq=freq) for l in lengths]
    index = ["Latest Standard Deviation {} Month".format(int(l/freq)*12) for l in lengths] # in terms of month
    

    return pd.Series(ann_stds,index=index,name=portfolio_return.name)

def calculate_rolling_returns(portfolio_return,month_size,freq):
    '''
    Calculate rolling returns over the given window size for the portfolio
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    month_size: int, window length in terms of month
    freq: int, frequency of portfolio
    
    Returns
    -----------------
    output: series, rolling returns start from the earliest possible date
    
    '''
    name = '{} {}-Month Rolling Return'.format(portfolio_return.name,month_size)
    
    window = int(month_size/12*freq)
    rolling_ret = portfolio_return.rolling(window).apply(lambda x: np.prod(x+1)-1,raw=True)
    rolling_ret.name = name
    return rolling_ret

def calculate_calendar_year_returns(portfolio_return,freq):
    '''
    Calculate calendar-year return.
    Calendar year time index is defined as the following:
        {t(y), all t in year y; y: all calendar years}
    Given y, {t(y)} is the portfolio return for the year y
    only length({t(y)}) >= freq, annualized return for year y would be computed
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    freq: int, frequency of portfolio
    
    Returns
    -----------------
    output: series, calendar year returns
    '''

    cal_ret = portfolio_return.groupby([lambda idx: idx.year]).apply(lambda x: np.prod(x+1)-1 if len(x)>=freq else np.nan)
    return cal_ret

def calculate_percentile(portfolio_return,pos):
    '''
    Calculate percentiles of portfolio return
    
    Parameters:
    -----------
    portfolio_return: pandas series, portfolio return
    pos: float or list of float
    
    Returns:
    --------
    float/numpy array
    '''
    return np.percentile(portfolio_return,pos)

def calculate_momentum(portfolio_return, N):
    '''
    Calculate momentum for portfolio return from N-1 periods till current time 
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    N: int, number of periods 
    
    Returns
    -----------------
    float, momentum based on N
    '''
    return np.prod(portfolio_return.tail(N)+1)-1

def calculate_return_risk_ratio(portfolio_return,freq):
    '''
    Calculate return risk ratio, annualized return of portfolio return and its annualized portfolio standard deviation
    
    Parameters:
    -----------
    portfolio_return: pandas series, portfolio return
    freq: freq: int, frequency of portfolio
    
    Return:
    -------
    float,return risk ratio
    '''
    return calculate_annualized_return(portfolio_return,freq)/calculate_annualized_stdev(portfolio_return,freq)

def calculate_skewness(portfolio_return):
    '''
    Calculate skewness of portfolio return
    
    Parameters:
    -----------
    portfolio_return: pandas series, portfolio return
    
    Returns:
    --------
    float, skewness of portfolio return
    '''
    return st.skew(portfolio_return,nan_policy='omit') # skewness

def calculate_kurtosis(portfolio_return):
    '''
    Calculate skewness of portfolio return
    
    Parameters:
    -----------
    portfolio_return: pandas series, portfolio return
    
    Returns:
    --------
    float, skewness of portfolio return
    '''
    return st.kurtosis(portfolio_return,nan_policy='omit') # kurtosis

def calculate_VaR_CVaR(portfolio_return,method='norm_CVaR',alpha=0.05):
    '''
    Calculate VaR and CVaR based on approximation with normal distribution and t distribution
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    alpha: float, 1-alpha means the VaR/CVaR level, by default 0.05
    
    Returns
    -----------------
    series, 4 values for t_VaR,norm_VaR,t_CVaR,norm_CVaR
    
    '''
    mu_n, sig_n = st.norm.fit(portfolio_return)
    nu, mu_t, sig_t = st.t.fit(portfolio_return)
    t_VaR = abs(st.t.ppf(alpha,df=nu,loc=mu_t,scale=sig_t))
    norm_VaR = abs(st.norm.ppf(alpha,loc=mu_n,scale=sig_n))
    t_CVaR = (-1/alpha)*st.t.expect(args=(nu,),loc=mu_t,scale=sig_t,lb=-np.inf,ub=-t_VaR)
    norm_CVaR =(-1/alpha)*st.norm.expect(loc=mu_n,scale=sig_n,lb=-np.inf,ub=-norm_VaR)
    result = dict(zip(['t_VaR','norm_VaR','t_CVaR','norm_CVaR'],[t_VaR, norm_VaR, t_CVaR, norm_CVaR]))
    if method is None:
        return result
    else:
        return result[method]

def calculate_drawdown(portfolio_return):
    '''
    Calculate drawdown over the entire history of portfolio return
    drawdown(t) = max(observed drawdown, change from peak(0,t) to time t)
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    
    Returns
    -----------------
    output: series, drawdown over the entire portfolio history
    
    '''
    cum_ret = np.cumprod(1+portfolio_return) # total return over the history
    drawdown_hist = pd.Series(index=portfolio_return.index.copy(),name=portfolio_return.name)   
    drawdown = 0
    max_seen = 1 
    
    for t in drawdown_hist.index:
            max_seen = max(max_seen,cum_ret[t]) # value of peak
            drawdown = max(drawdown,1-cum_ret[t]/max_seen) # compare the current observed drawdown with change from peak till time t
            drawdown_hist[t] = drawdown # update drawdown
    return drawdown_hist

def calculate_max_drawdown(portfolio_return):
    '''
    Calculate max drawdown over the entire history of portfolio return
    drawdown(t) = max(observed drawdown, change from peak(0,t) to time t)
    max_drawdown = max({drawdown(t): all t})
    
    Parameters
    -----------------
    portfolio_return: pandas series, portfolio return
    
    Returns
    -----------------
    float
    
    '''
    
    return np.max(calculate_drawdown(portfolio_return))

def calculate_batting_avg(target,benchmark):
    '''
    Calculate batting average value for target portfolio against benchmark portfolio
    
    M = # of time points when target return > benchmark return
    N = # of all time points
    batting average = M/N
    
    Parameters
    -----------------
    target: series, portfolio return
    benchmark: series, portfolio return
    
    Return
    -----------------
    float: batting average belonging to interval (0,1)
    
    '''
    excess_ret = np.subtract(target,benchmark)
    return np.sum(excess_ret>0)/np.sum(excess_ret.notna())

def calculate_beta(target,benchmark,rf):
    '''
    Calculate beta of excess returns, using OLS method, between target portfolio and benchmark portfolio
    
    target - rf = alpha + beta * (benchmark - rf) + error
    
    Parameters
    -----------------
    target: series, portfolio return
    benchmark: series, portfolio return
    rf: float/int/series, risk-free rate
    
    Returns
    -----------------
    float: beta
    
    '''
    target_excess = np.subtract(target, rf)
    benchmark_excess = np.subtract(benchmark, rf)
    
    return np.cov(target_excess,benchmark_excess,ddof=0)[0,1]/np.var(benchmark_excess,ddof=0)

def calculate_alpha(target,benchmark,rf):
    '''
    Calculate alpha of excess returns, using OLS method, between target portfolio and benchmark portfolio
    
    target - rf = alpha + beta * (benchmark - rf) + error
    
    Parameters
    -----------------
    target: series, portfolio return
    benchmark: series, portfolio return
    rf: float/int/series, risk-free rate
    freq: int, frequency of portfolio
    
    Returns
    -----------------
    float: alpha
    
    '''
    
    target_excess = np.subtract(target,rf)
    benchmark_excess = np.subtract(benchmark,rf)
    
    beta = calculate_beta(target,benchmark,rf)
    alpha = np.mean(np.subtract(target_excess,beta*benchmark_excess))
    
    return alpha #*freq # annualized alpha by simply multiplying the frequency of the portfolio(this is a estiamted method)

def calculate_treynor_ratio(target,benchmark,rf,freq):
    '''
    Calculate Treynor Ratio(TR) based on beta of excess returns
    
    TR = (annualized_return(target) - annualized_return(risk free))/beta(target,benchmark)
    
    Parameters
    -----------------
    target: series, portfolio return
    benchmark: series, portfolio return
    rf: float/int/series, risk-free rate, if it is a scalar which should be a annualized risk free rate
    freq: int,frequency of the portfolio return
    
    Returns
    -----------------
    float: Treynor Ratio
    
    '''
       
    ann_return_target = calculate_annualized_return(target,freq)
    if type(rf) is pd.Series:
        ann_return_rf = calculate_annualized_return(rf,freq)
    else:
        ann_return_rf = rf
    return (ann_return_target - ann_return_rf)/calculate_beta(target,benchmark,rf)

def calculate_sortino_ratio(target,benchmark):
    '''
    Calculate Sortino Ratio(SR), using normal distribution to approximate target portfolio return
    ***Notice that the SR is not based on annual frequency but the frequency of target portfolio
    
    SR = (R-T)/DR
        * R : average realized return of target portfolio
          T : required rate of return , defined by average realized return of benchmark portfolio
          DR : downside deviation E[(r-T)^2] on interval (-inf,T)
          r : random variable representing target portfolio return with normal distribution approximation
    
    Parameters
    -----------------
    target: series, portfolio return
    benchmark: series, portfolio return
    
    Returns
    -----------------
    float: Sortino Ratio
    '''
    
    R = target.mean()
    T = benchmark.mean()
    mu, sig = st.norm.fit(target[1:]) # first date contains no return 
    DR = np.sqrt(st.norm.expect(func=lambda x: (x-T)**2, loc=mu,scale=sig,lb=-np.inf,ub=T))
    return (R-T)/DR

def calculate_information_ratio(target,benchmark,freq):
    '''
    Calculate Information Ratio(IR) for annualized excess returns
    
    IR = R/S
        * R : annualized return of excess returns
          S : annualized standard deviation of excess returns
          excess returns: target_portfolio_return(t) - benchmark_portfolio_return(t)
          
    Parameters
    -----------------
    target: series, portfolio return
    benchmark: series, portfolio return
    freq: int,frequency of the portfolio return
    
    Returns
    -----------------
    float: Information Ratio
          
    '''
    
    ex = np.subtract(target , benchmark)
    return calculate_annualized_return(ex,freq)/calculate_annualized_stdev(ex,freq)


_portfolio_return_statistics_func_mapper = \
{'Annualized Return':calculate_annualized_return, # scalar  
 'Latest Annualized Return':calculate_latest_annualized_returns, # pandas series
 'Annualized Standard Deviation':calculate_annualized_stdev, # scalar
 'Latest Annualized Standard Deviation':calculate_latest_annualized_stdev, # pandas series
 'Month Rolling Return':calculate_rolling_returns, # pandas series
 'Calendar Year Return':calculate_calendar_year_returns, # pandas series
 'Percentile 10': calculate_percentile, # scalar
 'Percentile 25': calculate_percentile, # scalar
 'Percentile 50': calculate_percentile, # scalar
 'Percentile 75': calculate_percentile, # scalar
 'Percentile 90': calculate_percentile, # scalar
 'Momentum':calculate_momentum, # scalar
 'Return Risk Ratio':calculate_return_risk_ratio, # scalar
 'Skewness':calculate_skewness, # scalar
 'Kurtosis':calculate_kurtosis, # scalar
 'CVaR':calculate_VaR_CVaR, # scalar,(with method = 'norm_CVaR')
 'Drawdown':calculate_drawdown, # pandas series
 'Max Drawdown':calculate_max_drawdown, # scalar
 'Batting Average':calculate_batting_avg, # scalar
 'Beta':calculate_beta, # scalar
 'Alpha':calculate_alpha, # scalar
 'Treynor Ratio':calculate_treynor_ratio, # scalar
 'Sortino Ratio':calculate_sortino_ratio, # scalar
 'Information Ratio':calculate_information_ratio} # scalar

def get_portfolio_return_statistics_func_mapper(func_mapper = _portfolio_return_statistics_func_mapper):
    return func_mapper

def get_portfolio_return_statistics_func_list(func_mapper = _portfolio_return_statistics_func_mapper):
    return list(func_mapper.keys())

def get_portfolio_return_statistics_param_mapper(portfolio_return=None,freq=None,month_size=None,N=None,benchmark=None,rf=0):
    portfolio_return_statistics_param_mapper = \
    {'Annualized Return':{'portfolio_return':portfolio_return,'freq':freq},# method = 'direct'\
     'Latest Annualized Return':{'portfolio_return':portfolio_return,'freq':freq},\
     'Annualized Standard Deviation':{'portfolio_return':portfolio_return,'freq':freq}, # ddof = 1\
     'Latest Annualized Standard Deviation':{'portfolio_return':portfolio_return,'freq':freq},\
     'Month Rolling Return':{'portfolio_return':portfolio_return,'freq':freq,'month_size':month_size},\
     'Calendar Year Return':{'portfolio_return':portfolio_return,'freq':freq},\
     'Percentile 10':{'portfolio_return':portfolio_return,'pos':10},\
     'Percentile 25':{'portfolio_return':portfolio_return,'pos':25},\
     'Percentile 50':{'portfolio_return':portfolio_return,'pos':50},\
     'Percentile 75':{'portfolio_return':portfolio_return,'pos':75},\
     'Percentile 90':{'portfolio_return':portfolio_return,'pos':90},\
     'Momentum':{'portfolio_return':portfolio_return,'N':N},\
     'Return Risk Ratio':{'portfolio_return':portfolio_return,'freq':freq},\
     'Skewness':{'portfolio_return':portfolio_return},\
     'Kurtosis':{'portfolio_return':portfolio_return},\
     'CVaR':{'portfolio_return':portfolio_return}, # method = 'norm_CVaR', alpha = 0.05\ 
     'Drawdown':{'portfolio_return':portfolio_return},\
     'Max Drawdown':{'portfolio_return':portfolio_return},\
     'Batting Average':{'target':portfolio_return,'benchmark':benchmark},\
     'Beta':{'target':portfolio_return,'benchmark':benchmark,'rf':rf},\
     'Alpha':{'target':portfolio_return,'benchmark':benchmark,'rf':rf},\
     'Treynor Ratio':{'target':portfolio_return,'benchmark':benchmark,'rf':rf,'freq':freq},\
     'Sortino Ratio':{'target':portfolio_return,'benchmark':benchmark},\
     'Information Ratio':{'target':portfolio_return,'benchmark':benchmark,'freq':freq}}
    
    return portfolio_return_statistics_param_mapper



def calculate_statistics_of_interest(func_mapper,param_mapper,func_of_interest):
    
    def param_parse(param_mapper):
        old_new_dict = {}
        for f in func_of_interest:
            params_dict = param_mapper[f]
            params_name = params_dict.keys()
            params_name_modify = [pn for pn in params_name if pn not in ('portfolio_return','freq','target','rf','pos')]
            
            for pnm in params_name_modify:
                if np.isscalar(param_mapper[f][pnm]):
                    new_info = str(param_mapper[f][pnm])
                    old_new_dict[f] = '{} {}'.format(new_info,f)
                else:
                    new_info = str(param_mapper[f][pnm].name)
                    old_new_dict[f] = '{} vs {}'.format(f,new_info)
            
        return old_new_dict
    
    result = dict(zip(func_of_interest,[func_mapper[f](**param_mapper[f]) for f in func_of_interest]))
    
    old_new_dict = param_parse(param_mapper)
    
    for item in old_new_dict.items():
        result[item[1]] = result.pop(item[0])
    return result


def calculate_common_statistics(portfolio_return,freq,N=None):
    '''
    Calculate common statistics(only scalar result) that can be computed with a single portfolio return(no benchmark)
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    freq: int, frequency of portfolio
    N: int, used to calculate momentum from N+1 period before
    
    Returns
    -----------------
    series, with name of the regarding strategy and
            index as the statistics' name
    
    '''
    if N is None:
        N = portfolio_return.size
    
    common_param_mapper=get_portfolio_return_statistics_param_mapper(portfolio_return=portfolio_return,freq=freq,N=N)
    common_func_list = ['Annualized Return','Annualized Standard Deviation','Skewness','Kurtosis','Return Risk Ratio','CVaR','Momentum',\
                        'Max Drawdown','Percentile 10','Percentile 25','Percentile 50','Percentile 75','Percentile 90']
    func_mapper = get_portfolio_return_statistics_func_mapper()
    common_stats = calculate_statistics_of_interest(func_mapper,common_param_mapper,common_func_list)
    
    return pd.Series(common_stats,name=portfolio_return.name)

def calculate_scalar_statistics(portfolio_return,freq,N=None,benchmark=None,rf=0,only_benchmark=True):
    '''
    Calculate common statistics(only scalar result) that can be computed with a single portfolio return(no benchmark)
    
    Parameters
    -----------------
    portfolio_return: series, portfolio return
    freq: int, frequency of portfolio
    N: int, used to calculate momentum from N+1 period before
    benchmark
    
    Returns
    -----------------
    series, with name of the regarding strategy and
            index as the statistics' name
    
    '''
    common_statistics = calculate_common_statistics(portfolio_return,freq,N)
    if benchmark is None:
        return common_statistics
    else:
        other_param_mapper=get_portfolio_return_statistics_param_mapper(portfolio_return=portfolio_return,freq=freq,N=N,benchmark=benchmark,rf=rf)
        other_func_list = ['Batting Average','Beta','Treynor Ratio','Sortino Ratio','Information Ratio']
        func_mapper = get_portfolio_return_statistics_func_mapper()
        other_stats = calculate_statistics_of_interest(func_mapper,other_param_mapper,other_func_list)
        other_stats = pd.Series(other_stats,name=portfolio_return.name)
        if only_benchmark:
            return other_stats
        else:
            return pd.concat([common_statistics,other_stats],axis=0)

def calculate_all(weights,returns,name,freq,month_size,N):
    '''
    Calculate all information that can be computed with a strategy history and corresponding assets' returns
    and a few other necessary inputs
    
    Parameters
    -----------------
    weights: dataframe, strategy history 
    returns: dataframe, corresponding assets return history
        ***: weights and returns have same column names and date index
    name: str, name of the strategy
    freq: int, frequency of the portfolio
    month_size: int, window length to compute rolling returns for portfolio return, in terms of month
    N: int, number of periods to compute momentum for portfolio return
    
    Returns
    -----------------
    output: dict, keys for name and values for regarding information
    '''
    
    output = {}
    portfolio_return = calculate_portfolio_return(weights,returns,name)
    #print(any_portfolio)
    output['Return'] = portfolio_return # strategy/portfolio return history
    output['{}-Month Rolling Return'.format(month_size)] = calculate_rolling_returns(portfolio_return,month_size,freq)
    output['Calendar Year Return'] = calculate_calendar_year_returns(portfolio_return,freq)
    output['Drawdown'] = calculate_drawdown(portfolio_return)
    output["Statistics"] = calculate_statistics(portfolio_return,freq,N)
    output["Annualized Return"] = calculate_latest_annualized_returns(portfolio_return,freq)
    output["Annualized Standard Deviation"] = calculate_latest_annualized_stdev(portfolio_return,freq)

    return output