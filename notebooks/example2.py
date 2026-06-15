# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:08.028325Z","iopub.execute_input":"2024-05-02T02:35:08.029189Z","iopub.status.idle":"2024-05-02T02:35:12.426235Z","shell.execute_reply.started":"2024-05-02T02:35:08.029130Z","shell.execute_reply":"2024-05-02T02:35:12.424962Z"}}
import sys
from pathlib import Path
import subprocess
import os
import gc
from glob import glob

import joblib

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
# from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

ROOT = '/kaggle/input/home-credit-credit-risk-model-stability'

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.428946Z","iopub.execute_input":"2024-05-02T02:35:12.429574Z","iopub.status.idle":"2024-05-02T02:35:12.447807Z","shell.execute_reply.started":"2024-05-02T02:35:12.429528Z","shell.execute_reply":"2024-05-02T02:35:12.446446Z"}}
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.449568Z","iopub.execute_input":"2024-05-02T02:35:12.450235Z","iopub.status.idle":"2024-05-02T02:35:12.475478Z","shell.execute_reply.started":"2024-05-02T02:35:12.450195Z","shell.execute_reply":"2024-05-02T02:35:12.474396Z"}}
class Pipeline:
    @staticmethod
    def set_table_dtypes(df): #Standardize the dtype.
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))            

        return df
    
    @staticmethod
    def handle_dates(df): #Change the feature for D to the difference in days from date_decision.
        for col in df.columns:
            if (col[-1] in ("D",)) and ('count' not in col):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                
        df = df.drop("date_decision", "MONTH")

        return df
    
    @staticmethod
    def filter_cols(df): #Remove those with an average is_null exceeding 0.95 and those that do not fall within the range 1 < nunique < 200.
        for col in df.columns:
            # if col in ["decision_month", "decision_weekday"]:
            #     df = df.drop(col)
            #     continue
            # if ('amtde' in col) or ('bureau_b2' in col): # for ohter option
            #     continue

            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            # if '_depth2_' in col:
            #     continue
            if (col not in ["target", "case_id", "WEEK_NUM", ]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 50):#50 #len(df) * 0.20): # 95 # fe4 down at fq20
                    df = df.drop(col)
            
            # eliminate yaer, month feature
            # 644
            if (col[-1] not in ["P", "A", "L", "M"]) and (('month_' in col) or ('year_' in col)):# or ('num_group' in col):
            # if (('month_' in col) or ('year_' in col)):# or ('num_group' in col):
                df = df.drop(col)

        return df


# 644 lb best
class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_{col}") for col in cols]
        # expr_3 = [pl.median(col).alias(f"median_{col}") for col in cols]
        # expr_3 = [pl.var(col).alias(f"var_{col}") for col in cols]+ [pl.sum(col).alias(f"sum_{col}") for col in cols]
        # expr_3 = [pl.last(col).alias(f"last_{col}") for col in cols] #+ \
        #     [pl.first(col).alias(f"first_{col}") for col in cols] + \
        #     [pl.mean(col).alias(f"mean_{col}") for col in cols] + \
        #     [pl.std(col).alias(f"std_{col}") for col in cols]
        # expr_3 = [pl.count(col).alias(f"count_{col}") for col in cols]

        cols2 = [col for col in df.columns if col[-1] in ("L", "A")]
        expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2] + \
            [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2] # + \
            # [pl.first(col).alias(f"first_{col}") for col in cols2] + [pl.last(col).alias(f"last_{col}") for col in cols2]
        
        # BAD
        # cols3 = [col for col in df.columns if col[-1] in ("A")]
        # expr_4 = [pl.col(col).fill_null(strategy="zero").apply(lambda x: x.max() - x.min()).alias(f"max-min_gap_{col}") 
        #           for col in cols3]
        return expr_1 + expr_2 + expr_3 # + [pl.col(col).diff().last().alias(f"diff-last_{col}") for col in cols3] # + expr_4
    
    @staticmethod
    def applprev2_exprs(df):
        cols = [col for col in df.columns if "num_group" not in col]
        # expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] 
        expr_2 = [pl.first(col).alias(f"first_{col}") for col in cols]#  + [pl.last(col).alias(f"last_{col}") for col in cols]
        return []#expr_2

    @staticmethod
    def bureau_a1(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_{col}") for col in cols]

        cols2 = [
            # bad
        'annualeffectiverate_199L', 'annualeffectiverate_63L',
        'contractsum_5085717L', 
        'credlmt_230A', 'credlmt_935A',
        # 'debtoutstand_525A', 'debtoverdue_47A', 'dpdmax_139P', 'dpdmax_757P',
    #    'instlamount_768A', 'instlamount_852A',
    #    'interestrate_508L', 'monthlyinstlamount_332A',
    #    'monthlyinstlamount_674A', 
            # good?
       'nominalrate_281L', 'nominalrate_498L',
       'numberofcontrsvalue_258L', 'numberofcontrsvalue_358L',
       'numberofinstls_229L', 'numberofinstls_320L',
       'numberofoutstandinstls_520L', 'numberofoutstandinstls_59L',
       'numberofoverdueinstlmax_1039L', 'numberofoverdueinstlmax_1151L',
       'numberofoverdueinstls_725L', 'numberofoverdueinstls_834L',
            # bad?
    #    'outstandingamount_354A', 'outstandingamount_362A', 'overdueamount_31A',
    #    'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2_398A',
    #    'overdueamountmax_155A', 'overdueamountmax_35A',
        # bad ?
    #    'periodicityofpmts_1102L', 'periodicityofpmts_837L',
    #    'prolongationcount_1120L', 'prolongationcount_599L',
        # 520?
    #    'residualamount_488A', 'residualamount_856A', 'totalamount_6A',
    #    'totalamount_996A', 'totaldebtoverduevalue_178A',
    #    'totaldebtoverduevalue_718A', 'totaloutstanddebtvalue_39A',
    #    'totaloutstanddebtvalue_668A',
       ]

        # .697
        # expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2]
        
        # .696
        # expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2]

        # .697
        # expr_3 = [pl.std(col).alias(f"std_{col}") for col in cols2]
        
        # .6985
        # expr_3 = [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2]

        # .696
        # expr_3 = [pl.sum(col).alias(f"sum_{col}") for col in cols2] 

        # .6981
        # expr_3 = [pl.median(col).alias(f"median_{col}") for col in cols2]

        # .696
        # expr_3 = [pl.first(col).alias(f"first_{col}") for col in cols2] + [pl.last(col).alias(f"last_{col}") for col in cols2] # + \
        
        # .696
        # expr_3 = [pl.std(col).alias(f"std_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2]

        # .699
        # expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2] + \
        #     [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2]

        expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2] + \
            [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2] + \
            [pl.first(col).alias(f"first_{col}") for col in cols2] # + [pl.last(col).alias(f"last_{col}") for col in cols2] # not applied
        
        

        # expr_3 = [pl.col(col).fill_null(strategy="zero").apply(lambda x: x.max() - x.min()).alias(f"max-min_gap_depth2_{col}") for col in cols2]
        return expr_1 + expr_2 + expr_3    


    @staticmethod
    def bureau_b1(df):  # 0.95에서 미적용 중 # 36500
        # cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        # expr_1 = [pl.max(col).alias(f"bureau_b1_max_{col}") for col in cols]
        # expr_2 = [pl.min(col).alias(f"bureau_b1_min_{col}") for col in cols]

        # return expr_1 + expr_2 #  + expr_3
        return []
    
    
    @staticmethod
    def bureau_b2(df):  # 0.95에서 미적용 중 # 36500
        # cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        # expr_1 = [pl.max(col).alias(f"bureau_b2_max_{col}") for col in cols]
        # expr_2 = [pl.min(col).alias(f"bureau_b2_min_{col}") for col in cols]

        # return expr_1 + expr_2 #  + expr_3
        return []


    @staticmethod
    def deposit_exprs(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] # + \
            # [pl.last(col).alias(f"last_{col}") for col in cols]
            # [pl.mean(col).alias(f"mean_{col}") for col in cols] # + \
            # [pl.std(col).alias(f"std_{col}") for col in cols]  + \
             
            # [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_2 = [pl.first('openingdate_857D').alias(f'first_openingdate_857D')] + [pl.last('openingdate_857D').alias(f'last_openingdate_857D')]
        
        return expr_1 # + expr_2 #+ expr_ngmax

    @staticmethod
    def debitcard_exprs(df):
        # cols = [col for col in df.columns if (col[-1] in ["A"])]
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] 
            # [pl.mean(col).alias(f"mean_{col}") for col in cols] + \
            # [pl.std(col).alias(f"std_{col}") for col in cols]
        # expr_2 = [pl.first('openingdate_857D').alias(f'first_openingdate_857D')] + [pl.last('openingdate_857D').alias(f'last_openingdate_857D')]
        
        return expr_1 # + expr_2 #+ expr_ngmax
        # return expr_1


    @staticmethod
    def person_expr(df):
        cols1 = ['empl_employedtotal_800L', 'empl_employedfrom_271D', 'empl_industry_691L', 
                 'familystate_447L', 'incometype_1044T', 'sex_738L', 'housetype_905L', 'housingtype_772L',
                 'isreference_387L', 'birth_259D', ]
        # cols1 = [col for col in df.columns]
        expr_1 = [pl.first(col).alias(f"first_{col}") for col in cols1]
        
        expr_2 = [pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"), 
                  pl.col("mainoccupationinc_384A").filter(pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")]
        
        # No Effect ...
        # cols = ['personindex_1023L', 'persontype_1072L', 'persontype_792L']
        # expr_3 = [pl.col(col).last().alias(f"last_{col}") for col in cols] + [pl.col(col).drop_nulls().mean().alias(f"mean_{col}") for col in cols]

        # cols2 = [col for col in df.columns if col not in cols1]
        # expr_4 = [pl.max(col).alias(f"max_{col}") for col in cols2] + [pl.min(col).alias(f"min_{col}") for col in cols2] #  good at cv, bad at lb ?
            # [pl.col(col).drop_nulls().last().alias(f"last_{col}") for col in cols2] + [pl.col(col).drop_nulls().first().alias(f"first_{col}") for col in cols2] # no effect

        return expr_1 + expr_2 # + expr_4 # + expr_3
    
    @staticmethod
    def person_2_expr(df):
        # cols = [col for col in df.columns]
        cols = ['empls_economicalst_849M', 'empls_employedfrom_796D', 'empls_employer_name_740M'] # + \
            # ['relatedpersons_role_762T', 'conts_role_79M']
            # ['addres_district_368M', 'addres_role_871L', 'addres_zip_823M']

        expr_1 = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_2 = [pl.last(col).alias(f"last_{col}") for col in cols]

        # BAD
        # expr_ngc = [pl.count("num_group2").alias(f"count_num_group2")]
        # cols2 = [col for col in df.columns if (col in ("num_group1", "num_group2"))]
        # expr_ngmax = [pl.min(col).alias(f"min_{col}") for col in cols2] + [pl.max(col).alias(f"max_{col}") for col in cols2]

        # cols2 = [col for col in df.columns if col not in cols]
        # # expr_3 = [pl.max(col).alias(f"max_{col}") for col in cols2] + [pl.min(col).alias(f"min_{col}") for col in cols2] # no effect
        # expr_3 = [pl.col(col).drop_nulls().last().alias(f"last_{col}") for col in cols2] # no effect

        return expr_1 + expr_2 # + expr_3# + expr_ngc 

    @staticmethod
    def other_expr(df):
        expr_1 = [pl.first(col).alias(f"__other_{col}") for col in df.columns if ('num_group' not in col) and (col != 'case_id')]
        # cols1 = ['amtdepositbalance_4809441A', 'amtdepositincoming_4809444A', 'amtdepositoutgoing_4809442A']
        # expr_1 = [pl.last(col).alias(f"last_{col}") for col in cols1]
        # cols2 = ['amtdebitincoming_4809443A', 'amtdebitoutgoing_4809440A']
        # expr_3 = [(pl.col('amtdebitincoming_4809443A') - pl.col('amtdebitoutgoing_4809440A')).alias('amtdebit_incoming-outgoing')]
        return expr_1 # + expr_2 + expr_3
    
    
    @staticmethod
    def tax_a_exprs(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] + \
            [pl.last(col).alias(f"last_{col}") for col in cols] + \
            [pl.first(col).alias(f"first_{col}") for col in cols] + \
            [pl.mean(col).alias(f"mean_{col}") for col in cols] + \
            [pl.std(col).alias(f"std_{col}") for col in cols]
        # expr_1 = [pl.max(col).alias(f"max_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', 'num_group1']] + \
        #     [pl.min(col).alias(f"min_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', ]] + \
        #     [pl.mean(col).alias(f"mean_{col}") for col in ['amount_4527230A']] + \
        #     [pl.std(col).alias(f"std_{col}") for col in ['amount_4527230A']] + \
        #     [pl.last(col).alias(f"last_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', 'name_4527232M']] + \
        #     [pl.first(col).alias(f"first_{col}") for col in ['amount_4527230A', 'recorddate_4527225D', 'name_4527232M']] # BAD?

        expr_4 = [pl.col(col).fill_null(strategy="zero").apply(lambda x: x.max() - x.min()).alias(f"max-min_gap_depth2_{col}") for col in ['amount_4527230A']]

        return expr_1 + expr_4


    @staticmethod
    def bureau_a2(df): # 122만
        # cols = ['collater_valueofguarantee_1124L', 'pmts_dpd_1073P', 'pmts_overdue_1140A',]
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        expr_1 = [pl.max(col).alias(f"max_depth2_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_depth2_{col}") for col in cols]
        expr_3 = [pl.mean(col).alias(f"mean_depth2_{col}") for col in cols] + \
            [pl.std(col).alias(f"std_{col}") for col in cols]
        # expr_ngs = [pl.max(col).alias(f"max_{col}") for col in ['num_group1', 'num_group2', ]]

        expr_4 = [pl.col(col).fill_null(strategy="zero").apply(lambda x: x.max() - x.min()).alias(f"max-min_gap_depth2_{col}") for col in ['collater_valueofguarantee_1124L', 'pmts_dpd_1073P', 'pmts_overdue_1140A',]]

        expr_ngc = [pl.count("num_group2").alias(f"count_depth2_a2_num_group2")]

        # expr_5 = [pl.last(col).alias(f"last_{col}") for col in cols] + \
        #     [pl.first(col).alias(f"first_{col}") for col in cols] + \
        #     [pl.std(col).alias(f"std_{col}") for col in cols]

        return expr_1 + expr_2 + expr_3 + expr_4 + expr_ngc # + expr_5
    
    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df)

        return exprs

    
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.546131Z","iopub.execute_input":"2024-05-02T02:35:12.547239Z","iopub.status.idle":"2024-05-02T02:35:12.573617Z","shell.execute_reply.started":"2024-05-02T02:35:12.547191Z","shell.execute_reply":"2024-05-02T02:35:12.572323Z"}}
def agg_by_case(path, df):
    path = str(path)
    if '_applprev_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))

#     elif '_applprev_2' in path:
#         df = df.group_by("case_id").agg(Aggregator.applprev2_exprs(df))

    elif '_credit_bureau_a_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.bureau_a1(df))

    elif '_credit_bureau_b_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.bureau_b1(df))

    elif '_deposit_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.deposit_exprs(df))
    elif '_debitcard_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.debitcard_exprs(df))
        
    elif '_tax_registry_a' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.tax_a_exprs(df))
    elif '_tax_registry_b' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))
    elif '_tax_registry_c' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.get_exprs(df))
        
    elif '_other_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.other_expr(df))
    elif '_person_1' in path:
        df = df.sort("num_group1").group_by("case_id").agg(Aggregator.person_expr(df))
    elif '_person_2' in path:
        df = df.group_by("case_id").agg(Aggregator.person_2_expr(df))

    elif '_credit_bureau_a_2' in path:
        df = df.group_by("case_id").agg(Aggregator.bureau_a2(df))
    elif '_credit_bureau_b_2' in path:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    
    return df

def read_file(path, depth=None): 
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    
    if depth in [1, 2]:
        df = agg_by_case(path, df)
    
    return df

def read_files(regex_path, depth=None):
    print(regex_path)
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = agg_by_case(path, df)
        chunks.append(df)
        #     del df
        #     gc.collect()
        #     print('delete chunk')
        
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    
    return df

def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base.with_columns(
            decision_month = pl.col("date_decision").dt.month(),
            decision_weekday = pl.col("date_decision").dt.weekday(),
        )
    )
        
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
        
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    print(df_data.info())
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
        # cat_cols = [c for c in cat_cols if 'diff_' not in c]
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data, cat_cols

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.575798Z","iopub.execute_input":"2024-05-02T02:35:12.576686Z","iopub.status.idle":"2024-05-02T02:35:12.593361Z","shell.execute_reply.started":"2024-05-02T02:35:12.576632Z","shell.execute_reply":"2024-05-02T02:35:12.591807Z"}}
ROOT            = Path("/kaggle/input/home-credit-credit-risk-model-stability")

TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.595304Z","iopub.execute_input":"2024-05-02T02:35:12.596095Z","iopub.status.idle":"2024-05-02T02:35:12.606239Z","shell.execute_reply.started":"2024-05-02T02:35:12.596031Z","shell.execute_reply":"2024-05-02T02:35:12.604446Z"}}
# data_store = {
#     "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
#     "depth_0": [
#         read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
#         read_files(TRAIN_DIR / "train_static_0_*.parquet"),
        
#     ],
#     "depth_1": [
#         read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
#         read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
#         read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
#         # read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
#         # read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_person_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_other_1.parquet", 1),
#     ],
#     "depth_2": [
#         read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
#         read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
#     ]
# }

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.608596Z","iopub.execute_input":"2024-05-02T02:35:12.609262Z","iopub.status.idle":"2024-05-02T02:35:12.622765Z","shell.execute_reply.started":"2024-05-02T02:35:12.609219Z","shell.execute_reply":"2024-05-02T02:35:12.621598Z"}}
# df_train = feature_eng(**data_store)
# print("train data shape:\t", df_train.shape)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:12.624675Z","iopub.execute_input":"2024-05-02T02:35:12.625494Z","iopub.status.idle":"2024-05-02T02:35:13.332636Z","shell.execute_reply.started":"2024-05-02T02:35:12.625453Z","shell.execute_reply":"2024-05-02T02:35:13.330617Z"}}
data_store = {
    "df_base": read_file(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        read_file(TEST_DIR / "test_static_cb_0.parquet"),
        read_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
        read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
        read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
        read_file(TEST_DIR / "test_deposit_1.parquet", 1),
        read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
        read_file(TEST_DIR / "test_person_1.parquet", 1),
        read_file(TEST_DIR / "test_other_1.parquet", 1),
    ],
    "depth_2": [
        read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
        read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
        read_file(TEST_DIR / "test_person_2.parquet", 2),
    ]
}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:13.337192Z","iopub.execute_input":"2024-05-02T02:35:13.337534Z","iopub.status.idle":"2024-05-02T02:35:13.454908Z","shell.execute_reply.started":"2024-05-02T02:35:13.337506Z","shell.execute_reply":"2024-05-02T02:35:13.452272Z"}}
df_test = feature_eng(**data_store)
print("test data shape:\t", df_test.shape)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:13.456179Z","iopub.execute_input":"2024-05-02T02:35:13.457134Z","iopub.status.idle":"2024-05-02T02:35:16.140282Z","shell.execute_reply.started":"2024-05-02T02:35:13.457098Z","shell.execute_reply":"2024-05-02T02:35:16.139023Z"}}
# load models
cat_cols = joblib.load('/kaggle/input/hclgb-models/cat_cols.pickle')
ls_models = glob(os.path.join('/kaggle/input/hclgb-models', "*.pkl"))
models = [joblib.load(fn) for fn in ls_models]

lgb_features = models[0].feature_name_

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:16.141849Z","iopub.execute_input":"2024-05-02T02:35:16.142311Z","iopub.status.idle":"2024-05-02T02:35:16.154451Z","shell.execute_reply.started":"2024-05-02T02:35:16.142271Z","shell.execute_reply":"2024-05-02T02:35:16.152901Z"}}
# Drop the insignificant features
# df_train = df_train.pipe(Pipeline.filter_cols)
# df_test = df_test.select([col for col in df_train.columns if col != "target"])

df_test = df_test.select(['case_id'] + lgb_features)

# print("train data shape:\t", df_train.shape)
print("test data shape:\t", df_test.shape)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:16.156187Z","iopub.execute_input":"2024-05-02T02:35:16.156653Z","iopub.status.idle":"2024-05-02T02:35:16.268030Z","shell.execute_reply.started":"2024-05-02T02:35:16.156613Z","shell.execute_reply":"2024-05-02T02:35:16.266683Z"}}
del data_store
gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:16.269551Z","iopub.execute_input":"2024-05-02T02:35:16.270017Z","iopub.status.idle":"2024-05-02T02:35:16.713460Z","shell.execute_reply.started":"2024-05-02T02:35:16.269984Z","shell.execute_reply":"2024-05-02T02:35:16.712171Z"}}
df_test, _ = to_pandas(df_test, cat_cols)
df_test = reduce_mem_usage(df_test)
# display(df_test.head())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:16.715531Z","iopub.execute_input":"2024-05-02T02:35:16.716484Z","iopub.status.idle":"2024-05-02T02:35:16.873638Z","shell.execute_reply.started":"2024-05-02T02:35:16.716445Z","shell.execute_reply":"2024-05-02T02:35:16.872297Z"}}
df_test.drop(columns=["case_id"]).to_csv('test_data.csv', index=False)
del df_test
gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:16.877402Z","iopub.execute_input":"2024-05-02T02:35:16.877778Z","iopub.status.idle":"2024-05-02T02:35:16.882970Z","shell.execute_reply.started":"2024-05-02T02:35:16.877749Z","shell.execute_reply":"2024-05-02T02:35:16.881593Z"}}
# %reset -f
# !mkdir -p submission_data
# !mv test_data.csv submission_data/test_data.csv  # move file to submission_data dir
# import glob
# for f in glob.glob('*'):
#     if not f.startswith('submission_data'):# delete all files if path not starts with submission_data
#         !rm -rf {f}

# %% [markdown]
# ### Submission

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:16.884247Z","iopub.execute_input":"2024-05-02T02:35:16.884557Z","iopub.status.idle":"2024-05-02T02:35:17.128625Z","shell.execute_reply.started":"2024-05-02T02:35:16.884532Z","shell.execute_reply":"2024-05-02T02:35:17.127501Z"}}
import sys
from pathlib import Path
import subprocess
import os
import gc
from glob import glob

import joblib

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

ROOT = '/kaggle/input/home-credit-credit-risk-model-stability'

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:17.130422Z","iopub.execute_input":"2024-05-02T02:35:17.130832Z","iopub.status.idle":"2024-05-02T02:35:18.386427Z","shell.execute_reply.started":"2024-05-02T02:35:17.130724Z","shell.execute_reply":"2024-05-02T02:35:18.385160Z"}}
# load models
cat_cols = joblib.load('/kaggle/input/hclgb-models/cat_cols.pickle')
ls_models = glob(os.path.join('/kaggle/input/hclgb-models', "*.pkl"))
models = [joblib.load(fn) for fn in ls_models]

print(len(models), models)

lgb_features = models[0].feature_name_
len(lgb_features), lgb_features

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:18.387843Z","iopub.execute_input":"2024-05-02T02:35:18.388266Z","iopub.status.idle":"2024-05-02T02:35:18.393960Z","shell.execute_reply.started":"2024-05-02T02:35:18.388233Z","shell.execute_reply":"2024-05-02T02:35:18.392612Z"}}
def set_categoricals(df_data, cat_cols):
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:18.395446Z","iopub.execute_input":"2024-05-02T02:35:18.395875Z","iopub.status.idle":"2024-05-02T02:35:18.411450Z","shell.execute_reply.started":"2024-05-02T02:35:18.395837Z","shell.execute_reply":"2024-05-02T02:35:18.410136Z"}}
def lgb_prediction(feats, models):
    predictions = np.zeros(len(feats))
    for model in models:
        p = model.predict_proba(feats)[:, 1]
        predictions += p/len(models)
    return predictions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:18.414029Z","iopub.execute_input":"2024-05-02T02:35:18.414366Z","iopub.status.idle":"2024-05-02T02:35:18.988681Z","shell.execute_reply.started":"2024-05-02T02:35:18.414341Z","shell.execute_reply":"2024-05-02T02:35:18.987333Z"}}
CHUNK_SIZE = 10 ** 6
reader = pd.read_csv('/kaggle/working/test_data.csv', chunksize=CHUNK_SIZE)
y_pred = []
for df_chunk in reader:
    # p = predictor.predict_proba(df_chunk).iloc[:, 1].values
    df_chunk = set_categoricals(df_chunk, cat_cols)
    p = lgb_prediction(df_chunk, models)
    y_pred.append(p)
    
y_pred = np.concatenate(y_pred, axis=0)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:18.989932Z","iopub.execute_input":"2024-05-02T02:35:18.991126Z","iopub.status.idle":"2024-05-02T02:35:19.103530Z","shell.execute_reply.started":"2024-05-02T02:35:18.991087Z","shell.execute_reply":"2024-05-02T02:35:19.102143Z"}}
del reader
gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:19.104846Z","iopub.execute_input":"2024-05-02T02:35:19.105233Z","iopub.status.idle":"2024-05-02T02:35:19.122471Z","shell.execute_reply.started":"2024-05-02T02:35:19.105195Z","shell.execute_reply":"2024-05-02T02:35:19.121172Z"}}
df_subm = pd.read_csv(f"{ROOT}/sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = y_pred#lgb_pred

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-02T02:35:19.123943Z","iopub.execute_input":"2024-05-02T02:35:19.124599Z","iopub.status.idle":"2024-05-02T02:35:19.139836Z","shell.execute_reply.started":"2024-05-02T02:35:19.124567Z","shell.execute_reply":"2024-05-02T02:35:19.138811Z"}}
df_subm.to_csv("submission.csv")
print(df_subm)