import pandas as pd
import numpy as np
from xlwings.utils import rgb_to_int


def organize_result(result):
    copy = {}
    for k in result.keys():
        res = result[k].copy()
        if ('return' in k.lower() and 'annualized' not in k.lower()) or 'drawdown' in k.lower():
            res = res.T
        res.index.name = k
        if 'rolling return' in k.lower():
            res.index.name = str(res.isna().sum(axis=1).unique()[0]+1)+' '+res.index.name
        copy[k] = res
    return copy

def reporting(output_report,report_order,sheet):

    output_report = organize_result(output_report)

    init_top = 21
    init_left = 1

    top = init_top
    left = init_left
    sheet.range((top,left),(100,500)).clear()
    for s in report_order:

        position = (top,left)
        top_left_cell = sheet.range(position)
        top_left_cell.expand('table').clear()
        output = output_report[s]

        top_left_cell.value = output

        bottom_right_cell = top_left_cell.expand('table').last_cell
        bottom = bottom_right_cell.row
        right = bottom_right_cell.column

        if s == 'Portfolio Statistics':
            number_formatting(top_left_cell,'0.00%','0.00',False)
        elif s in ('Drawdown','Return','Month Rolling Return'):
            number_formatting(top_left_cell,'0.00%',date=True)
        elif s in ('Latest Annualized Return','Latest Annualized Standard Deviation','Calendar Year Return'):
            number_formatting(top_left_cell,'0.00%',date=False)
        else:
            number_formatting(top_left_cell,'0.00',date=False)

        top = bottom + 2
        left = left

def number_formatting(start_cell,general_format,special_format=None,date=True):
    start_cell.expand('table').number_format = general_format
    start_cell.expand('right').number_format = 'General'
    if date:
        start_cell.expand('right').number_format = 'yyyy-mm-dd;@'
    if special_format is not None:
        special_formatting(start_cell,special_format)

def special_formatting(start_cell,special_format):
    # Skewness, Kurtosis, Beta, and Ratio are the rows should be changed into '0.00' from '0.00%'
    format_label_dict = {'0.00':['Skewness','Kurtosis','Beta','Ratio']}
    format_label = format_label_dict[special_format]

    shape = start_cell.expand('table').shape
    row,col = start_cell.row,start_cell.column
    idx = pd.Index(start_cell.expand('down').value)
    pos = sum([idx.str.find(label) != -1 for label in format_label])
    pos = np.arange(len(pos))[pos==1] + start_cell.row
    sheet = start_cell.sheet
    for p in pos:
        p = int(p)
        sheet.range((p,col),(p,col+shape[1])).number_format = special_format


def refresh_report_sheet(wb):
    if 'Report' in [s.name for s in wb.sheets]:
        wb.sheets['Report'].delete()
    report_sheet = wb.sheets.add('Report')
    return report_sheet

class Report_Sheet:
    default_position = (5,1)
    cell_height = 14.4
    cell_width = 55.2

    def __init__(self,sheet,position=None):
        self.sheet = sheet
        if position is None:
            self.position = self.default_position
        else:
            self.position = position
        self.table_list = []
        self.chart_list = []
        self.range_list = []
        self.last_cell_list = []

    def __repr__(self):
        return 'Sheet Name:{}\n Table {}\n Chart {}\n Range {}'.\
                format(self.sheet.name,self.get_table_info(),self.get_chart_info(),self.get_range_info())

    def get_table_info(self):
        count = len(self.table_list)
        return '{}: {}'.format(count,[i.name for i in self.table_list])

    def get_chart_info(self):
        count = len(self.chart_list)
        return '{}: {}'.format(count,[i.name for i in self.chart_list])

    def get_range_info(self):
        count = len(self.range_list)
        return '{}: {}'.format(count,[i.name for i in self.range_list])

    def update_position(self,old_position):
        end_cell = self.last_cell_list[-1]
        row,col = end_cell.row,end_cell.column
        if col < 7:
            return (old_position[0],col + 2)
        else:
            max_row = max([i.row for i in self.last_cell_list])
            return (max_row+2,1)

    def display_basic_info(self):
        cell1= self.sheet.range('A1')
        cell1.value = 'Report'
        cell1.api.font.bold = True
        cell1.api.font.color = rgb_to_int((255,255,255))
        cell1.color = rgb_to_int((153,0,204))

        self.sheet.range('B1').formula = '=TODAY()'

    def insert_table(self,data,table_name,format_type,position=None,table_style = 'NatixisTableStyle',index=True,header=True,totalrow=True):
        format_type_dict = {'type1':('0.00%',None,False),
                            'type2':('0.00%',None,True),
                            'type3':('0.00%','0.00',False)}

        if position is None:
            position = self.position
        else:
            pass

        start_cell = self.sheet.range((position))
        start_cell.name = table_name.replace(' ','_')

        start_cell.options(index=index,header=header).value = data
        start_cell.sheet.activate()
        start_cell.select()
        table = start_cell.sheet.api.ListObjects.add()
        table.name = ('Table '+table_name).replace(' ','_')
        table.showautofilter = False
        table.TableStyle = table_style
        table.showtotals = totalrow
        number_formatting(start_cell,*format_type_dict[format_type])
        #sheet.range(position[0]-1,position[1]).value = table_name

        self.table_list.append(table)
        self.range_list.append(start_cell)
        self.last_cell_list.append(start_cell.expand('table').last_cell)
        self.position = self.update_position(position)

    def insert_chart(self,source_data,chart_type,switch_row_col,chart_title):
        chart = self.sheet.charts.add()
        chart.set_source_data(source_data)
        chart.chart_type = chart_type
        if switch_row_col == True:
            chart.api[1].plotby = 1
        else:
            chart.api[1].plotby = 2

        chart.left = source_data.left
        chart.top = source_data.top
        chart.width = min(max(500,source_data.width),1000)
        chart.height = min(source_data.height*3,400)

        chart.api[1].setelement(2)
        title_set = chart.api[1].charttitle
        title_set.text = chart_title
        title_set.font.bold = False
        title_set.font.color = rgb_to_int((153,0,255))

        chart.api[1].legend.font.size = 12

        chart.name = chart_title

        self.position = (int((chart.top + chart.height)/self.cell_height) + 3,1)
