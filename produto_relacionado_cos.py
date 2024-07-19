from nltk.corpus import stopwords
import pymysql.connections as sql
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import html
from sklearn.metrics.pairwise import cosine_similarity
import sqlalchemy
import pymysql.cursors as cursors
from datetime import datetime
from urllib.parse import quote
import numpy as np
from math import ceil

# Tira HTML da descricao
def cleanhtml(desc):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', desc)
  return cleantext

languages = {'espanhol': 'spanish ', 'finlandês': 'finnish ', 'francês': 'french ', 'inglês': 'english ', 'português': 'portuguese '}

conn = sql.Connection()
livros = pd.read_sql(f"""SELECT sku,sku_correspondente, nome,descricao_curta,descricao,autor, idioma_nome,ids_categorias FROM produto
LEFT JOIN (SELECT sku, GROUP_CONCAT(DISTINCT IFNULL(filtered.id_categoria_subs,id_categoria) ORDER BY IFNULL(filtered.id_categoria_subs,id_categoria)) AS ids_categorias FROM categoria_produto 
INNER JOIN tocalivros_analise.categoria_filtrada AS filtered USING (id_categoria)
GROUP BY sku) cats USING (sku)
LEFT JOIN produto_idioma USING (id_idioma)
WHERE `status` = 'ativo' AND produto.sku LIKE '1%'
ORDER BY sku""",conn)
conn.close()

# Ajuste de descricao e outras strings para facilitar os cálculos
livros['desc_ajust'] = livros['descricao'].fillna(livros['descricao_curta'])
livros['desc_ajust'] = livros['desc_ajust'].fillna('')
livros['autor'] = livros['autor'].fillna('')
livros['autor'] = livros['autor'].str.strip()
livros['nome'] = livros['nome'].str.strip()
livros['idioma_nome'] = livros['idioma_nome'].str.split(' ').str.get(0).str.lower().str.strip()
livros['desc_ajust'] = livros['desc_ajust'].apply(html.unescape)
livros['desc_ajust'] = livros['desc_ajust'].apply(cleanhtml)
livros['desc_ajust'] = livros['desc_ajust'].str.strip().str.lower()
livros['ids_categorias'] = livros['ids_categorias'].fillna('')
livros = livros.drop(livros.index[(livros['ids_categorias'].isna()) & (livros['desc_ajust'] == '')], inplace = False).reset_index(drop=True)
livros = livros.drop(livros.index[(~livros['idioma_nome'].isin(languages.keys())) & (livros['ids_categorias'].isna())], inplace = False).reset_index(drop=True)
# drop_list = livros.loc[livros['sku'].str.len() == 7,'sku_correspondente'].dropna(inplace = False).index.tolist()

def get_similarity(itens,language,len_group = 5000):
    """Função para pegar entre produtos da mesma lingua utilizando o método de similaridade de cossenos
    itens: DataFrame contendo os produtos
    language: Língua (em inglês e.g. spanish)
    len_group: Tamanho do bloco que será processado, onde maiores valores resultarão numa velocidade de processamento maior ao custo de mais memória"""
    audios = itens[itens['sku'].str.len() == 6].index
    ebooks = itens[itens['sku'].str.len() == 7].index
    recom = pd.DataFrame({'sku_de':[],'sku_para':[],'posicao':[]})
    vectorizer = TfidfVectorizer(analyzer='word',
                    ngram_range=(1, 2),
                    min_df=2,
                    # max_df=0.5,
                    #  max_features = 5000,
                    stop_words=stopwords.words(language))
    tfidf_matrix = vectorizer.fit_transform(itens['nome'] + " " + itens['desc_ajust'])
    vectorizer_number = TfidfVectorizer(analyzer='word',
                        ngram_range=(1, 2),
                        min_df=2,
                        # max_df=0.5,
                        #  max_features = 5000
                        )
    tfidf_matrix_number = vectorizer_number.fit_transform(itens['ids_categorias'])
    for x in range(ceil(itens.shape[0]/len_group)):
        cos_desc = cosine_similarity(tfidf_matrix[x*len_group:(x+1)*len_group],tfidf_matrix)
        cos_cat = cosine_similarity(tfidf_matrix_number[x*len_group:(x+1)*len_group],tfidf_matrix_number)
        for y in range(len(cos_desc)):
            id_real = y+x*len_group
            sim_desc = cos_desc[y]*0.85
            sim_cat = cos_cat[y]*0.15
            res = sim_desc + sim_cat
            same_author = itens[(itens['autor'] == itens.loc[id_real,'autor']) & (itens.index != id_real)].index
            if len(same_author) > 1:
                res[same_author] = res[same_author]+0.05 #Adicionando score para livros do mesmo autor
            res[id_real] = 0 # Tirar o score do livro para não cair nas recomendações
            validos = np.where((res > 0.075) & (res < 1))
            # Tirar multiplos livros do mesmo ator
            if np.isin(same_author,audios).sum() > 2:
                res[same_author[np.argpartition(res[same_author[np.isin(same_author,audios)]],-2)[:-2]]] = 0
            if np.isin(same_author,ebooks).sum() > 2:
                res[same_author[np.argpartition(res[same_author[np.isin(same_author,ebooks)]],-2)[:-2]]] = 0

            res_audios = res.copy()
            res_audios[ebooks] = 0
            res_ebooks = res.copy()
            res_ebooks[audios] = 0

            final = np.array([])

            if id_real in audios:
                for _ in range(8):
                    ind = np.argpartition(res_audios,-3)[-3:]
                    ind = ind[np.isin(ind,validos)]
                    final = np.append(final,ind[np.argsort(res_audios[ind])[::-1]])
                    res_audios[ind] = 0
                    ind = np.argpartition(res_ebooks,-2)[-2:]
                    ind = ind[np.isin(ind,validos)]
                    final = np.append(final,ind[np.argsort(res_ebooks[ind])[::-1]])
                    res_ebooks[ind] = 0
                    if len(final) >= 15:
                        break
            else:
                for _ in range(8):
                    ind = np.argpartition(res_ebooks,-3)[-3:]
                    ind = ind[np.isin(ind,validos)]
                    final = np.append(final,ind[np.argsort(res_ebooks[ind])[::-1]])
                    res_ebooks[ind] = 0
                    ind = np.argpartition(res_audios,-2)[-2:]
                    ind = ind[np.isin(ind,validos)]
                    final = np.append(final,ind[np.argsort(res_audios[ind])[::-1]])
                    res_audios[ind] = 0
                    if len(final) >= 15:
                        break
            res = itens.loc[final[:15],['sku']].reset_index(drop = True)
            res['posicao'] = res.index + 100
            res['sku_de'] = itens.loc[id_real,'sku']
            res['sku_para'] = res['sku'] 
            recom = pd.concat([recom,res[['sku_de','sku_para','posicao']]])
    return recom

sugg = pd.DataFrame({'sku_de':[],'sku_para':[],'posicao':[]})
for lingua, language in languages.items():
    sugg = pd.concat([sugg,get_similarity(livros[livros['idioma_nome'] == lingua].reset_index(drop = True), language)])
sugg['id_produto_relacionado_tipo'] = 1
sugg = sugg[sugg['sku_de'].isin(sugg.groupby('sku_de').size()[sugg.groupby('sku_de').size() >= 3].index)]

# Pega a base antiga para manter os itens adicionados manualmente
conn = sql.Connection()
og = pd.read_sql("SELECT sku_de, sku_para, id_produto_relacionado_tipo, posicao ,updated_at FROM produto_relacionado",conn)
manual = og[og['posicao'] < 100].copy()
og = og[og['posicao'] >= 100].copy()
cursor = cursors.Cursor(conn)
cursor.execute("TRUNCATE TABLE produto_relacionado")
conn.commit()
conn.close()
manual = manual[(manual['sku_de'].isin(livros['sku'])) & (manual['sku_para'].isin(livros['sku']))].copy()
final = sugg.merge(og, how = 'left', on = ['sku_de','sku_para','posicao'], suffixes = ('','_2'))
final['updated_at'].fillna(pd.to_datetime(int(datetime.now().timestamp()) - 10800, unit = 's'), inplace = True)
final['id_produto_relacionado_tipo'].fillna(1, inplace = True)
sug = pd.concat([manual,final[['sku_de','sku_para','id_produto_relacionado_tipo','posicao','updated_at']]]).sort_values(by = ['sku_de','posicao'], ascending = [True,True])
sug.drop_duplicates(subset = ['sku_de','sku_para'],inplace = True)
sug['id_produto_relacionado'] = range(1,sug.shape[0]+1)
sug['posicao'] = sug['posicao'].astype(int)

# Append na tabela produto_relacionado
pasw = quote()
try:
    engine = sqlalchemy.create_engine()
    sug[['sku_de','sku_para','id_produto_relacionado_tipo','posicao','updated_at']].to_sql('produto_relacionado', con = engine, if_exists = 'append',index = False)
    engine.dispose()
except Exception:
    import traceback
    import os
    log_file = os.path.expanduser("~\Desktop\error_produto_relacionado.log")
    with open(log_file,'a') as file:
        file.write('\n'+str(datetime.now())+':\n'+str(traceback.format_exc()))