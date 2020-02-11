
##########
# File: adjust_datetime.py
# Description:
#    Coleção de funções para ajustar/converter datetime
##########


import pendulum
import datetime


#Funcao para corrigir o numero da semana  do mes (1:5)
def week_of_month(data):
  dt = pendulum.parse(str(data))
  return dt.week_of_month

