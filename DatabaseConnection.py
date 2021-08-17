import sqlite3
import pandas as pd

class DatabaseConnection(object):
    
    def DBup(self, databaseFile = 'data/moniotr.db'):
        self.databaseFile = databaseFile
        self.conn = sqlite3.connect(self.databaseFile)
        self.cur = self.conn.cursor()
        return self
    
    
    def DBdown(self):
        self.cur.close()
        return self
        
        
    def dictionaryCursor(self, boolean):
        if boolean:
            self.conn.row_factory = sqlite3.Row
            self.cur = self.conn.cursor()
        else: 
            self.conn.row_factory = None
            self.cur = self.conn.cursor()
            
            
    def executeSql(self, sql, data = None):
        if data:
            res = self.cur.execute(sql, data)
        else:
            res = self.cur.execute(sql)
            
        if "DELETE" or "UPDATE" or "INSERT" in sql:
            self.conn.commit()
            
        return res

    def exportCsv(self, sql, databaseFile = '/opt/moniotr/moniotr.db'):
        self.databaseFile = databaseFile
        conn = sqlite3.connect(self.databaseFile, isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)
        dbDf = pd.read_sql_query(sql, conn)
        dbDf = dbDf.loc[:,~dbDf.columns.duplicated()] # drop duplicate columns
        csv = dbDf.to_csv(index=False)
        return csv
    

if __name__ == "__main__":
    dbconn = DatabaseConnection()
    dbconn.DBup('tests/moniotr.db.test')
    dbconn.dictionaryCursor(True)
    assert(dbconn.conn.row_factory == sqlite3.Row)
    dbconn.dictionaryCursor(False)
    assert(dbconn.conn.row_factory == None)
    dbconn.DBdown()
    
