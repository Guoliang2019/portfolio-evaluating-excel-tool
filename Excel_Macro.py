import pandas as pd
import xlwings as xw
import numpy as np
from Excel_Reporting_Help import *
from Python_Workflow import main,clean_up
from additional_help import process_row_col,find_strategy_benchmark
import win32api

@xw.sub
def Macro_Description():
    wb = xw.Book.caller()
    info = '''
    Tool Intro: Introduction to Portfolio Evaluation Tool\n
    Setup: Set up Portfolio Evaluation sheet for Portfolio Information and Customized Benchmark Setting\n
    Portfolio Evaluate: Computing return/risk metrics\n
    Report: Generate quality report for one strategy and one benchmark\n
    Clear: Clear up Portfolio Evaluation sheet
    '''
    win32api.MessageBox(wb.app.hwnd, info,'Instruction')

@xw.sub
def Marco_Clear():
    wb = xw.Book.caller()
    wb.sheets['Portfolio Evaluation'].clear()
    if 'Report' in [i.name for i in wb.sheets]:
        wb.sheets['Report'].delete()

@xw.sub
def Macro_Setup():
    wb = xw.Book.caller()
    
    
    sheet = wb.sheets['Portfolio Evaluation']
    
    sheet.range((1,1),(20,12)).clear_contents()
    
    content = {1:'Portfolio Name',2:'Portfolio Frequency(1Y ; 4Q ; 12M ; 52W ; 251D)',3:'Rolling Window Size(Month)',\
               4:'Momentum N (Period)',5:'Cash Excluded '}
    for row in range(5):
        row = row + 1
        rng = sheet.range((row,1))
        rng.value = content[row]
    
    portfolio_name = pd.Index(wb.sheets['Fund Weights'].range('A1').expand('right').value).dropna().unique().values
    sheet.range('B1').expand('right').value = portfolio_name
    
    indecies = pd.Index(wb.sheets['Benchmark Returns'].range('A1').expand('right').value).dropna().unique().values
    bdf = pd.DataFrame(index=indecies,columns=['B'+str(i) for i in range(1,5)])
    bdf.index.name = 'Benchmark Constituents'
    
    sheet.range((9,1)).value = bdf
    sheet.range((9,1)).api.font.bold=True
    
    sheet.range((1,1),(20,12)).api.style = 'Input'
    sheet.range((1,1)).expand('down').api.font.bold = True
    sheet.range((9,1)).api.font.bold = True
    sheet.range((2,2),(5,2)).number_format = 'General'
    win32api.MessageBox(wb.app.hwnd, "Please fill in necessary information within the colored region",'Notice')

@xw.sub
def Macro_Portfolio_Evaluating():
    wb = xw.Book.caller()
    work_book = wb.name
    
    #settable_inputs = pd.read_excel(work_book,sheet_name='Portfolio Evaluation',header=None,index_col=None)
    settable_inputs = wb.sheets['Portfolio Evaluation'].range((1,1),(20,20)).options(convert=pd.DataFrame,index=False,header=False).value
    #fund_weights = pd.read_excel(work_book,sheet_name = 'Fund Weights',header=[0,1],index_col=0)
    fund_weights = wb.sheets['Fund Weights'].range('A1').options(expand='table',convert=pd.DataFrame,header=2,index=1).value
    #custom_returns = pd.read_excel(work_book,sheet_name = 'Custom Returns',header=0,index_col=0)
    custom_returns = wb.sheets['Custom Returns'].range('A1').options(expand='table',convert=pd.DataFrame,header=1,index=1).value
    #equity_returns = pd.read_excel(work_book,sheet_name = 'Equity Returns',header=0,index_col=0)
    equity_returns = wb.sheets['Equity Returns'].range('A1').options(expand='table',convert=pd.DataFrame,header=1,index=1).value
    #benchmark_returns = pd.read_excel(work_book,sheet_name = 'Benchmark Returns',header=0,index_col=0)
    benchmark_returns = wb.sheets['Benchmark Returns'].range('A1').options(expand='table',convert=pd.DataFrame,header=1,index=1).value
    
    result = main(settable_inputs,fund_weights,custom_returns,equity_returns,benchmark_returns)
    
    report_sheet = wb.sheets['Portfolio Evaluation']
    report_order = ['Portfolio Statistics','Latest Annualized Return','Latest Annualized Standard Deviation',\
                    'Correlation','Calendar Year Return','Month Rolling Return','Return','Drawdown']
    #report_sheet.range((21,1)).value = 'testing'
    reporting(result,report_order,report_sheet)
    
@xw.sub
def Macro_Reporting():
    wb = xw.Book.caller()
    
    Portfolio_Name = wb.api.Application.InputBox('Strategy Name').strip()
    Benchmark_Name = wb.api.Application.InputBox('Benchmark Name').strip()
    
    settable_inputs = wb.sheets['Portfolio Evaluation'].range((1,1),(20,20)).options(convert=pd.DataFrame,index=False,header=False).value
    fund_weights = wb.sheets['Fund Weights'].range('A1').options(expand='table',convert=pd.DataFrame,header=2,index=1).value
    custom_returns = wb.sheets['Custom Returns'].range('A1').options(expand='table',convert=pd.DataFrame,header=1,index=1).value
    equity_returns = wb.sheets['Equity Returns'].range('A1').options(expand='table',convert=pd.DataFrame,header=1,index=1).value
    benchmark_returns = wb.sheets['Benchmark Returns'].range('A1').options(expand='table',convert=pd.DataFrame,header=1,index=1).value

    cleaned_data = clean_up(settable_inputs,fund_weights,custom_returns,equity_returns,benchmark_returns)
    
    fund_weights = cleaned_data['fund_weights']
    benchmark_weights = cleaned_data['benchmark_weights']
    
    ### retrieve relevant data
    portfolio_weight=fund_weights.tail(1).stack('Portfolio').stack().reset_index().rename(columns={0:'Weight'}).iloc[:,1:].set_index('Portfolio').loc[Portfolio_Name]
    benchmark_weight = benchmark_weights.tail(1).stack('Portfolio').stack().reset_index().rename(columns={0:'Weight'}).iloc[:,1:].set_index('Portfolio').loc[Benchmark_Name]
    stats = [i.refers_to_range for i in wb.names if i.name == 'Portfolio_Statistics'][0].expand().options(pd.DataFrame,index=1,header=1).value
    stats = find_strategy_benchmark(stats,Portfolio_Name,Benchmark_Name)
    
    sheet = refresh_report_sheet(wb)
    report_sheet = Report_Sheet(sheet)
    report_sheet.display_basic_info()
    report_sheet.insert_table(portfolio_weight,'Strategy','type1')
    report_sheet.insert_table(benchmark_weight,'Benchmark','type1')
    report_sheet.insert_table(fund_weights[Portfolio_Name].T,'Weight History','type2',totalrow=False)
    report_sheet.insert_chart(report_sheet.range_list[-1].expand(),'area_stacked',True,'Weight History')
    report_sheet.insert_table(stats,'Statistics','type3',totalrow=False)
    report_sheet.insert_chart(report_sheet.range_list[-1].expand(),'column_clustered',True,'Statistics')
