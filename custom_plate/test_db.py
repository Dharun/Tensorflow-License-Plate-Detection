#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:27:59 2018
    Script created to support tensorflow
    It pushes License plate value along with the timestamp
    in POSTGRESSQL
@author: tensorflow-cuda
"""
import psycopg2
def psyco_insert_plate(plate):
    plate_char = str(plate)
    try:
        connect_str = "dbname='testpython' user='postgres' host='localhost' " + \
                  "password='postgres'"
        conn = psycopg2.connect(connect_str)
        print('ok')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO testing (numplate) VALUES (%s)',(plate_char))
        conn.commit()
        print('ok')
        cursor.execute("""SELECT * from testing""")
        rows = cursor.fetchall()
#        print(rows)
        print('Plate inserted')
        return rows
    except Exception as e:
        print("Big problem. Invalid dbname, user or password?")
        print(e)
    
yo = 'sdffv'
print(psyco_insert_plate(yo))