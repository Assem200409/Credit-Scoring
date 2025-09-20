import pandas as pd
import numpy as np


class Preparing:
    """
    Класс для подготовки и фильтрации клиентов.
    """
    
    def __init__(self, df):
        self.df = df                                        # оригинал, если понадобится
        self.df_copy = df.copy()                            # создали копию, с которой будем работать 
        
    def rename_columns(self):
        self.df_copy.rename(
            columns={"default.payment.next.month": "default", 'PAY_0':'PAY_1'}, 
            inplace = True                                  # меняем сущ-ий датасет
        )

    def warning(self):
        print('Версия_5')
    
    def select_columns(self):
        """
        Выбираем колонки с оплатой, по которым будет производиться фильтрация.
        """
        self.rename_columns()
        self.pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        self.X_pay = self.df_copy.loc[:, self.pay_columns].values # преобразовываем в массив
        self.y = self.df_copy.default.values
        
    def group_clients(self):
        """
        'removed_0': False по умолчанию. Клиенты со своевременными платежами PAY_<=0 и с дефолтом (y == 1). 
        Присваивается значение True, если клиент попадает под это условие. Аномальная группа.
        'removed_1': False по умолчанию. Клиенты со своевременными платежами.
        self.index_no_d: клиенты, которые остались после фильтрации.
        self.index: клиенты, которые удовлетворяют условию ['removed_0'] = True (удаляем).
        self.index_no_d: индексы клиентов оставшихся после фильтрации.
        """
        self.select_columns()
        self.df_copy['removed_0'] = False
        self.df_copy['removed_1'] = False
        self.index = []                                             # список для помеченных клиентов (удаляем)
        self.index_no_d = []                                        # список для всех остальных клиентов
        for i,x in enumerate( self.X_pay ):                         # i - индекс клиента, x - одномерный массив PAY_0...PAY_6  
            if (len(np.where (x<=0)[0]) == 6 and self.y[i]==1):     # все шесть месяцев PAY_<=0 (ни одной просрочки в течение 6 мес.) => итог все равно дефолтным вышел (метка 1 присвоена)
                self.index.append(i)                                # записываем индекс клиента в список "удаленных"
                self.df_copy.loc[i, ['removed_0']] = True           # если клиент попадает под условие (все шесть месяцев PAY_<=0 и дефолт), то помечаем True флажком.      
            else:
                self.index_no_d.append(i)                           # все остальные в список не удаленных клиентов    

        self.df_new, self.X_pay_new, self.y_new = (                 
            self.df_copy.loc[self.index_no_d,:],                    # сохраняем датафрейм без аномалий (строка - непроблемные клиенты и все столбцы)
            self.X_pay[self.index_no_d,:],                          # дисциплинарные признаки без аномалий
            self.y[self.index_no_d] )                                # новый вектор меток без аномалий
        self.X_all_new = self.df_copy.loc[self.index_no_d, ['LIMIT_BAL', 'PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
           'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].values
        
        return self.df_copy.loc[self.index_no_d,:], self.X_pay[self.index_no_d,:], self.y[self.index_no_d] # возвращаем эти объекты
