import pandas as pd
from sqlalchemy import create_engine

class surveys:
    
    def __init__(self, database='/global/u2/l/lonappan/workspace/s4bird/s4bird/Data/surveys.db'):
        
        self.database = database
        self.engine = create_engine(f'sqlite:///{self.database}', echo=True)
        self.tables = self.engine.table_names()
        
    
    def get_table_dataframe(self,table):
        if table not in self.tables:
            raise ValueError(f"{table} not in {self.tables}")
        connection = self.engine.connect()
        df = pd.read_sql_table(table,connection)
        connection.close()
        return df
    
    def write_table_dic(self,dic,table):
        df = pd.DataFrame.from_dict(dic)
        connection = self.engine.connect()
        df.to_sql(table,connection)
        connection.close()
        

        