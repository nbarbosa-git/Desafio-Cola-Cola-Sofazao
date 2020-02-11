
##########
# File: adjust_datatype.py
# Description:
#    Coleção de funções para ajustar/converter tipos de dados
##########



#Funcao que converte numericos com milhar ',' e decimal '.', para o tipo numerico PT-BR
def PT_BR_string_to_numeric(df, col):
    df[col] = df[col].replace(to_replace=r'(?<=\d)[\.]', value='', regex=True)
    return  df[col].replace(to_replace=r'(?<=\d)[\,]', value='.', regex=True).astype(float)




