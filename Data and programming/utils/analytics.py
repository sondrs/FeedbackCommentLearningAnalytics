import pandas as pd
import numpy as np
from utils import nlp, prep, analytics
from tabulate import tabulate
from matplotlib import pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler

def importDataCSV(filename = 'Evaluation_2020_2021',
                path = 'Input/'):
    
    from utils import prep
    
    #importing data
    data = pd.read_csv(path+filename+'.csv', low_memory=False)
    data['KJØNN'] = [prep.fixK(v) for v in data.KJØNN]
    SEMESTER_remade = []
    for item in data.SEMESTER:
        i = prep.splitter(item)
        if len(i) == 2:
            if i[1] and i[0]:
                SEMESTER_remade.append((i[1])+prep.vhconverter(i[0]))
            else:
                SEMESTER_remade.append(item)
        else:
            SEMESTER_remade.append(item)


    data['SEMESTER_date'] = SEMESTER_remade
    data['semester'] = [re.sub('[^A-ZÆØÅa-zæøå]','',str(item)) for item in data.SEMESTER]
    data['year'] = [re.sub('[^0-9]','',str(item)) for item in data.SEMESTER]

    data['Answer complete'] = pd.to_datetime(data['Answer complete'])

    #using the informative categorical data as indices
    data = data.set_index(['SCHOOL',
                           'BY',
                           'STUDIEPROGRAM',
                           'Course',
                           'SEMESTER_date',
                           'KJØNN',
                           'ALDER',
                           'Respondent unique ID',
                           'Answer complete',
                           'Language']).drop(columns=['Answer created'])

    #identifying and assigning columns in a column MultiIndexaccording to their type
    #free text columns
    stringCols = ['Hvilke digitale aktiviteter dette semesteret bidro mest til ditt læringsutbytte?',
                    'Comments',
                    'Praksis : Dersom du har andre synspunkter på praksisperioden, skriv dem gjerne her.']
    stringColsGuide = dict(zip(stringCols,[';Digitalt læringsmiljø',
                       ';General',
                       '']))

    addedCols = ['semester', 'year', 'SEMESTER']
    addedColsGuide = dict(zip(addedCols,[';other', ';other', ';other']))

    boolCols = ['Har du hatt digital undervisning dette semesteret?']
    boolColsGuide = dict(zip(boolCols,['Digitalt læringsmiljø']))

    #all others are 1-5 scales
    intCols = data.columns.drop(stringCols+boolCols+addedCols)

    #cleaning the column names
    repeatedText = 'I hvor stor grad er du enig i følgende påstand... : ...'

    newCols = [prep.colFixer(col, repeatedText, 
                             stringCols, boolCols, addedCols,
                            stringColsGuide, addedColsGuide, boolColsGuide) for col in data.columns]
    newIndex = ['dtype','category','measure']

    #this one references the original column names
    addCols = pd.DataFrame([item.split(';') for item in newCols],
                 columns=['original']+newIndex).set_index('original').T

    #np.array([item.split(';') for item in newCols])
    multiCols = pd.MultiIndex.from_arrays(addCols.values, names=newIndex)

    data = pd.DataFrame(data.values, index=data.index, columns=multiCols)
    intShapes = data.loc[:,'int'].shape

    #forcing data into numerical values where applicable
    intData = pd.to_numeric(
        np.array([re.sub('[^0-9]','',str(item)) 
                for item in data.loc[:,'int'].values.flatten()])).reshape(intShapes)

    data.loc[:,'int'] = intData
    
    data = data.copy()

    data[('other','other','gotComment')] = [True if type(x)==str 
                                        else False 
                                        for x in data.string.General.Comments.tolist()]
    
    idxGen = (i for i in range(data.loc[data.other.other.gotComment==True].shape[0]))
    data[('other','other','docIdx')] = [prep.idxLabeler(c,idxGen) for c in 
                                        data.other.other.gotComment]
    
    #display(data.head(2))
    #print('shape:',data.shape)
    #print('comments:',len(data.string.General.Comments.dropna()))
    return data




def createCommentsDF(data):
    
    from utils.nlp import tokenizer
    
    data = data.copy()
    commentsDF = pd.DataFrame(data.loc[data.other.other.gotComment==True,
             ('string','General')].merge(
        data.loc[data.other.other.gotComment==True,
                ('other','other')],left_index=True,right_index=True).drop(
        columns=data.other.other.columns.drop(['docIdx'])).values,
                columns=['documents','docIdx']).convert_dtypes().set_index('docIdx')
    commentsDF = commentsDF.copy()
    
    commentsDF['documents'] = [(' ').join(tokenizer(doc)) for doc in commentsDF['documents']]
    
    #display(commentsDF.head(3))
    #print('shape:',commentsDF.shape)
    return commentsDF




def createSentencesDF(commentsDF):

    mapping = dict(zip(commentsDF.index
        ,
             [item.split(' . ') for item in commentsDF.documents]))
    docLengths = [len(doc) for doc in mapping.values()]
    allSentences = []
    for doc in mapping.values():
        allSentences+=doc
    docIdx = []
    for i,idx in enumerate(mapping.keys()):
        docIdx+=[idx]*docLengths[i]
    docIdx

    sentencesDF = pd.DataFrame({'docIdx':docIdx,
                  'sentence':[item for item in allSentences]})
    
    #display(sentencesDF.head(5))
    #print('shape:',sentencesDF.shape)
    
    return sentencesDF


def dfGrouper(DF):

    from utils import nlp
    DF = DF.reset_index()
    DF['sentences'] = [True]*DF.shape[0]
    for l in nlp.Values().validSentiments:
        DF[l] = [l in cell for cell in DF['sentiment_scored']]

    for l in nlp.Values().validTopics:
        DF[l] = [1/len(cell) if l in cell else 0 for cell in DF['topic_scored']]
        for p in nlp.Values().validSentiments:
            DF[l+'_'+p] = [1/len(DF['topic_scored'][i]) 
                           if l in DF['topic_scored'][i] 
                           and p in DF['sentiment_scored'][i] 
                           else 0 
                           for i in range(DF.shape[0])]


    DF = DF.groupby(by='docIdx').sum(numeric_only=True).drop(columns=['index']) 
    for l in nlp.Values().validSentiments:
        DF[l] = DF[l]/DF['sentences']
    for l in nlp.Values().validTopics:
        DF[l] = DF[l]/DF['sentences']
        DF[l+'_Positive'] = DF[l+'_Positive']/(DF['Positive']*DF['sentences'])
        DF[l+'_Negative'] = DF[l+'_Negative']/(DF['Negative']*DF['sentences'])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    DF['sentiment_scored'] = scaler.fit_transform((DF.Positive-DF.Negative).values.reshape(-1,1))
    for t in nlp.Values().validTopics:
        values = (DF[t+'_Positive']-DF[t+'_Negative']).values
        values[values==0]=np.nan
        DF[t+'_sentiment_scored'] = scaler.transform(values.reshape(-1,1))
        del values

    eligibles = (#(DF['sentences'].values>4).astype(int) 
                 #+ 
                 np.array([(DF.loc[i,'Positive']>0 
                            and 
                            DF.loc[i,'Negative']>0) 
                           for i in range(DF.shape[0])]).astype(int))
    
    eligibles = eligibles>0
    DF['constructiveness'] = [np.nan]*DF.shape[0]
    DF.loc[eligibles,'constructiveness'] = (- 2*(DF.loc[eligibles,'sentiment_scored']**2) + 1)
    
    dropCols = []
    for t in nlp.Values().validTopics:
        for s in nlp.Values().validSentiments:
            dropCols.append(t+'_'+s)
    for s in nlp.Values().validSentiments:
        dropCols.append(s)
    DF = DF.drop(columns=dropCols)
    #display(DF.head(3))
    return DF


def nlpScoredMerger(commentsScored,commentsDF,plot=False):
    from utils import nlp
    mIndex = pd.MultiIndex.from_tuples(
        [('nlp','metrics',w) for w in ['sentences','constructiveness']]
       +[('nlp','topics',col) for col in nlp.Values().validTopics]
       +[('nlp','sentiment',col) for col in ['sentiment_scored']+[w+'_sentiment_scored' for w in nlp.Values().validTopics]])
    newCommentsScored = pd.DataFrame(columns=mIndex)
    for idx in mIndex:
        newCommentsScored[idx]=commentsScored[idx[2]]
    commentsScored = newCommentsScored.copy()
    del newCommentsScored
    del mIndex
    
    if plot:
        fig, (p1,p2) = plt.subplots(1,2,figsize=(10,3),dpi=240,sharey=True)
        commentsScored.nlp.sentiment.sentiment_scored.plot.hist(bins=10,title='Distribution of sentiment_scored',ax=p1)
        p1.set_xlabel('level of satisfaction')
        p1.grid(True)
        for val in nlp.Values().validTopics:
            p2.bar(val, commentsScored[('nlp','topics',val)].sum(axis=0))

        p2.set_xticks(list(dict(enumerate(nlp.Values().validTopics)).keys()), labels=nlp.Values().validTopics,rotation='vertical')
        p2.set_title('Frequency of topics in comments')
        p2.grid(True)
        plt.show()
    
    commentsDF = pd.DataFrame(
        commentsDF.copy().documents.tolist(),
        columns=pd.MultiIndex.from_tuples([
            ('string','General','Comments')
        ])).rename_axis('docIdx')
    commentsDF = commentsDF.merge(commentsScored,left_index=True, right_index=True)
    #display(commentsDF.head(3))
    return commentsDF


def loadDataAndProcess(filename= 'Evaluation_2020_2021',
                path = 'Input/'):
    from utils import nlp, prep, analytics
    
    data = analytics.importDataCSV(filename = 'Evaluation_2020_2021',
                path = 'Input/')

    commentsDF = analytics.createCommentsDF(data)

    sentencesDF = analytics.createSentencesDF(commentsDF)
    vectorizer, svd_tfidf = nlp.loadVectorizerSVD()
    sentencesDF['sentencePreprocessed'] = nlp.preprocessFunction(sentencesDF.sentence)
    sentences_reduced_by_svd = svd_tfidf.transform(vectorizer.transform(sentencesDF.sentencePreprocessed))
    sentencesDF = sentencesDF.reset_index().merge(nlp.productionPreds(sentences_reduced_by_svd),
                                                  left_index=True,
                                                  right_index=True).set_index(['index']).drop(
        columns=['sentencePreprocessed'])

    commentsScored = analytics.dfGrouper(sentencesDF)
    commentsDF = analytics.nlpScoredMerger(commentsScored,commentsDF,plot=False)
    commentsDF['other','other','docIdx'] = commentsDF.index
    commentsDF = commentsDF.reset_index().drop(columns=['docIdx'])

    data = data.merge(commentsDF.drop(columns=[('string','General','Comments')]).set_index(('other','other','docIdx')), 
                how='left', left_on=[('other','other','docIdx')], right_index=True).copy()
    data.int = (data.int-1)/2-1
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    def scaleComp(val):
        if val>10:
            return 10
        else:
            return val
    #sentencenumScaled = np.array([scaleComp(val) for val in RobustScaler().fit_transform(
    #    data.nlp.metrics.sentences.values.reshape(-1,1)).ravel()])
    #print(sentencenumScaled)
    sentencenumScaled = np.array([scaleComp(val) for val in data.nlp.metrics.sentences.values])
    sentencenumScaled = MinMaxScaler(
        feature_range=(-1,1)).fit_transform(sentencenumScaled.reshape(-1,1)).ravel()
    
    data.loc[:,('nlp','metrics','sentences')] = sentencenumScaled
    
    return data

def analysisDashboard(filename = 'Evaluation_2020_2021',
                path = 'Input/'):
    
    from utils import analytics
    data = analytics.loadDataAndProcess(filename = filename,
                                        path = path)
    

    from jupyter_dash import JupyterDash
    import dash
    from dash import Dash, dash_table
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas
    import numpy as np
    import matplotlib.colors as mcolors
    from pandas.api.types import is_string_dtype
    from pandas.api.types import is_numeric_dtype
    pd=pandas

    indexDict = {0:'SCHOOL',1:'BY',2:'STUDIEPROGRAM',3:'COURSE',4:'SEMESTER_date',5:'KJØNN',6:'ALDER','NO':'Not aggregated'}
    dropdownstyler={'font-family':'Helvetica', 'font-size':14, 'color':'#2A3F5F'}

    metadataTypeOptions = [{'label': html.Span([indexDict[idx]], style=dropdownstyler), 
                       'value': idx} for idx in range(0,7)]
    checkliststyler={'font-family':'Helvetica','color':'#2A3F5F', 'font-size':16, 'padding-right':7}
    aggregateoptions=(metadataTypeOptions+
    [{'label': html.Span(['Do not aggregate'], style=dropdownstyler),'value':'NO'}])
    
    filterParameterValues = ['nlp/'+('/').join(list(i)) for i in data.nlp.columns if i[0] in  ['metrics','sentiment']]+[('/').join(m) for m in [
        ['int','Emnet','Hvor tilfreds er du, alt i alt, med emnet?'],
        ['int','Underviser(ne)','underviser(ne) bidro positivt til min læring og hjalp meg med å nå målene mine med dette emnet'],
        ['int','Eksamen','undervisningen gjorde at jeg var godt forberedt til eksamensarbeidet'],
        ['int','Digitalt læringsmiljø','jeg opplevde underviser som tilstedeværende i det digitale læringsmiljøet']]]
    filterParameterOptions = [{'label': html.Span([('    /    ').join(m.split('/'))], style=dropdownstyler),
                    'value': m} for m in filterParameterValues]

    metricvalues = [('/').join(list(i)) for i in data.columns]
    metricoptions= [{'label': html.Span([('    /    ').join(m.split('/'))], style=dropdownstyler),
                    'value': m} for m in metricvalues]
    app = JupyterDash(__name__)


    app.layout = html.Div([
        #logo and title
        html.Div([
            html.Div([html.Img(src='assets/Kristiania_symbol.jpg',width=50)],style={'width':'3%',
                                                                                   'display':'inline-block',
                                                                                   'padding-right':10}),
            html.Div([html.H1('Course Evaluation Analysis App')], style={'font-family':'Helvetica',
                                                                        'width':'89%',
                                                                        'display':'inline-block',
                                                                        'padding-bottom':0}
                    )
        ],style={'width':'99%','display':'inline-block'}
        ),

        #line
        html.Hr(),

        #METADATA FILTERS

        html.Div([
            html.Div([html.H4('Metadata filters:')
                           ],style={'font-family':'Helvetica',
                                                                         'color':'#2A3F5F',
                                                                        'width':'99%',
                                                              'display':'inline-block'}),



            html.Div([
                dcc.Dropdown(
                    id='metadataTypeSelector',
                    options=metadataTypeOptions,  
                    value=metadataTypeOptions[0]['value'],
                    clearable=False)
            ],style={'width':'25%','display':'inline-block'}
            ),
            html.Div([
                dcc.Dropdown(
                    id='metadataSelector', 
                    value=[],
                    clearable=True,
                    multi=True
                )],style={'width':'70%','display':'inline-block','padding-bottom':0, 'padding-right':0}
            ),
            html.Div([
                dcc.Dropdown(
                    id='metadataTypeSelector1',
                    options=[], 
                    value=[],
                    clearable=True)
            ],style={'width':'25%','display':'inline-block'}
            ),
            html.Div([
                dcc.Dropdown(
                    id='metadataSelector1', 
                    value=[],
                    clearable=True,
                    multi=True
                )],style={'width':'70%','display':'inline-block','padding-bottom':0, 'padding-right':0}
            ),
            #SELECT METRICS FOR CHART
            html.Div([
                html.Div([
                    html.H4('Select metrics:')
                ],style={'font-family':'Helvetica',
                         'color':'#2A3F5F'}
                ),

                #metrics
                html.Div([
                    dcc.Dropdown(
                            id='metrics',
                            options=metricoptions,
                            value=[],
                            clearable=True,
                            multi=True
                    )
                    ],style={'width':'96%','display':'inline-block', 'padding-top':0}
            )      
            ],style={'width':'99%','display':'inline-block','padding-right':0, 'padding-top':0}
            )  
        ], style={'width':'57%','display':'inline-block'}
        ),
        

        #AGGREGATE ON-OPTIONS
        html.Div([
            html.Div([
                html.H4('Aggregate on:')
            ],style={'width':'99%','display':'inline-block','font-family':'Helvetica',
                                                                         'color':'#2A3F5F'}
            ),
            html.Div([
                dcc.Dropdown(
                    id='aggregator',
                    options=aggregateoptions,
                    value=aggregateoptions[-1]['value'], 
                    clearable=False)
            ],style={'width':'99%','display':'inline-block'}
            ),

            #DATA FILTERS
            html.Div([
                html.Div([
                    html.Span([html.H4('Data filters:')], style={'font-family':'Helvetica','color':'#2A3F5F', 'padding-left':0, 'padding-bottom':0})
                ], style={'width':'99%','display':'inline-block', 'padding-left':0, 'padding-bottom':0}
                ),
                html.Div([
                    dcc.Dropdown(
                    id='filterParameter',
                    options=filterParameterOptions,
                    value=[],
                    clearable=True)            
                ], style={'width':'33%','display':'inline-block'}
                ),
                html.Div([
                    dcc.RangeSlider(
                        -1, 1, marks=None,
                       value=[-1,1],
                       id='filterParameterSlider',
                       tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'font-family':'Helvetica','width':'66%','display':'inline-block'}
                ),


            ],style={'width':'99%','display':'inline-block'}
            )

        ],style={'width':'42%','display':'inline-block'}
        ),

        html.Hr(),




        #CHARTS
        html.Div([
            html.Div([
                dcc.Graph(id='bar-chart')
            ],style={'width':'50%','display':'inline-block'}
            ),
            html.Div([
                dcc.Graph(id='hist')
            ],style={'width':'45%','display':'inline-block'}
            )
        ]
        ),
        html.Hr(),

        #DATA TABLE
        dash_table.DataTable(
                             id='data-table',
                             style_table={'height': '300px', 'overflowY': 'auto'},
                             style_data={'whiteSpace':'normal',
                                        'height':'auto',
                                        'lineHeight':'15px',
                                        'font-family':'Helvetica'},
                             style_header={'whiteSpace':'normal',
                                        'height':'auto',
                                        'lineHeight':'20px',
                                        'font-family':'Helvetica',
                                          'font-weight':'Bold'},
                             style_cell={'textAlign':'left',
                                        'maxWidth':400},

                            page_size=20)
    ], style={'backgroundColor':'white'})


    #INTERACTIVITY 
    @app.callback(
        [Output('bar-chart','figure'),
        Output('data-table', 'data'),
        Output('metadataSelector','options'),
        Output('hist','figure'),

        Output('metadataTypeSelector1','options'),
        Output('metadataSelector1','options'),
         Output('metadataTypeSelector1','clearable')
        ],


        [Input('metadataTypeSelector','value')
            ,Input('metadataSelector','value')
         ,Input('aggregator','value')
        ,Input('metrics','value')
        ,Input('filterParameter','value')
        ,Input('filterParameterSlider','value')

         ,Input('metadataTypeSelector1','value')
         ,Input('metadataSelector1','value')
         
               ]
    )


    #FIG/CONTENT UPDATER
    def update_content(metadataTypeSelector,metadataSelector,lev,metrics,filterParameter,filterParameterSlider,
                      metadataTypeSelector1,metadataSelector1,
                      ):
        bars = go.Figure()
        hist = go.Figure()
        aggregateType = 'mean' #The way the data has been prepped, mean is the interesting value in every use case



        #IMPLEMENTING METADATA FILTERING
        locList = [metadataSelector if metadataTypeSelector==i else data.index.get_level_values(i).unique() for i in range(7)]
        if metadataSelector1:
            locList[metadataTypeSelector1] = metadataSelector1

        locTuple=tuple(locList)
        filtered = data.loc[locTuple]

        if metadataTypeSelector1:
            metadataSelectorOptions1 = [{'label': html.Span([idx], style=checkliststyler),
                                          'value': idx} for idx in filtered.index.get_level_values(metadataTypeSelector1).unique().dropna()]
            #metadataSelectorOptions1 = sorted(metadataSelectorOptions1)
        else:
            metadataSelectorOptions1 = []

        #IMPLEMENTING DATA VALUE FILTERING
        if filterParameter:

            filterParameter = filterParameter.split('/')
            filtered = filtered.loc[filterParameterSlider[0] <= data[filterParameter[0]][filterParameter[1]][filterParameter[2]]]
            filtered = filtered.loc[filterParameterSlider[1] >= data[filterParameter[0]][filterParameter[1]][filterParameter[2]]]
            filterParameterOptions1 = [i for i in filterParameterOptions if i['value']!=filterParameter]


        if metrics:
            if type(metrics)==str:
                metrics=[metrics.split('/')]
            if type(metrics)==list:
                if type(metrics[0])==str:
                    metrics = [m.split('/') for m in metrics]


            #IMPLEMENTING METRIC FILTERING
            filtered = pd.DataFrame({'Comments':filtered['string']['General']['Comments']}
                                    #|
                                    #{'gotComment':filtered['other']['other']['gotComment']}
                                |
                                {item[2]:filtered[item[0]][item[1]][item[2]] for item in metrics}
                                 )


            #COMPUTING MEANS
            for cf in metrics:
                if type(lev)==int:
                    subfiltered = filtered[cf[2]]#+['gotComment']]
                    if is_numeric_dtype(subfiltered):
                        means = subfiltered.groupby(level=lev).mean()
                        counts = subfiltered.groupby(level=lev).count()
                        sums = subfiltered.groupby(level=lev).sum()
                    else:
                        counts = subfiltered.groupby(level=lev).count()
                        means = pd.Series(np.array([np.nan]*counts.shape[0]))
                        sums = means
                else:
                    if lev == 'NO':
                        subfiltered = filtered[cf[2]]
                        if metadataSelector and filtered[cf[2]].dropna().shape[0]:
                            if is_numeric_dtype(subfiltered):
                                means = subfiltered.mean()
                                counts = subfiltered.count();
                                sums = subfiltered.sum()
                            else:
                                counts = subfiltered.count()
                                means = pd.Series(np.array([np.nan]*counts.shape[0]))
                                sums = means

                    else:
                        means=np.nan
                        sums=means
                        counts=means

                if aggregateType=='mean':
                    plotVals = means
                if aggregateType=='sum':
                    plotVals = sums
                if aggregateType=='count':
                    plotVals = counts

            #ADDING TO CHART
                if lev=='NO':
                    bars.add_trace(go.Bar(x=['Overall'], y=[plotVals], name='/'.join(cf))) 
                    hist.add_trace(go.Histogram(x=subfiltered,
                                           xbins={
                                           'start':-1,
                                           'end':1,
                                           'size':0.05},
                              name='/'.join(cf)))
                else:
                    bars.add_trace(go.Bar(x=plotVals.index, y=plotVals, name='/'.join(cf)))   
                    hist.add_trace(go.Histogram(x=subfiltered,
                                           xbins={
                                           'start':-1,
                                           'end':1,
                                           'size':0.05},
                              name='/'.join(cf)))
                bars.update_traces(marker_line_color = '#7F7F7F', marker_line_width = 1)

                

            if lev=='NO':
                aggregInfo = 'Overall'
            else:
                aggregInfo = 'Aggregated on '+indexDict[lev]


        else:
            filtered = pd.DataFrame({'Comments':filtered['string']['General']['Comments']})
            aggregInfo = 'N/A'
        bars.update_layout(barmode='group',



                             showlegend=True,
                             legend_groupclick='toggleitem',
                             legend_itemclick='toggle',
                             legend_itemdoubleclick='toggleothers',
                             legend_title_text='Metric(s):',
                             legend_orientation='h',
                             legend_x=0,
                             legend_y=1.3,

                             hovermode='closest',
                             hoverlabel_namelength=50,
                             hoverlabel_bgcolor='white',

                             hoverlabel_font_size=12,
                             hoverlabel_font_family='Helvetica',
                             clickmode='event+select',

                             font_family='Helvetica',
                             separators='. ',

                             plot_bgcolor=mcolors.CSS4_COLORS['whitesmoke'],
                             paper_bgcolor='#fff',
                             colorway=list(mcolors.TABLEAU_COLORS.values()),

                             xaxis_title= aggregInfo,
                             yaxis_title=aggregateType.capitalize()+' Value',

                             yaxis_range=(-1,1)
                           
                             

                            )

        hist.update_layout(
                xaxis_title_text='Overall', 
                yaxis_title_text='Count', 
                bargap=0.1, 
                bargroupgap=0.0,
                xaxis_range=(-1,1),
                colorway=list(mcolors.TABLEAU_COLORS.values()),
                hovermode='closest',
                hoverlabel_namelength=50,
                hoverlabel_bgcolor='white',
                hoverlabel_font_size=12,
                hoverlabel_font_family='Helvetica',
                clickmode='event+select',
                font_family='Helvetica',
                separators='. ',
                plot_bgcolor=mcolors.CSS4_COLORS['whitesmoke'],
                paper_bgcolor='#fff',
                showlegend=True,
                legend_groupclick='toggleitem',
                legend_itemclick='toggle',
                legend_itemdoubleclick='toggleothers',
                legend_title_text='Metric(s):',
                legend_orientation='h',
                legend_x=0,
                legend_y=1.3,
        ) 

        metadataSelectorOptions = [{'label': html.Span([idx], style=checkliststyler),
                                      'value': idx} for idx in data.index.get_level_values(metadataTypeSelector).unique().dropna()]


        dispCols = filtered.reset_index().columns.drop(['Respondent unique ID','Answer complete','Language']).tolist()
        dispCols = dispCols[:-len(metrics)][-1:]+dispCols[:-len(metrics)][:-1]


        metadataTypeSelectorOptions1 = [{'label': html.Span([indexDict[idx]], style=dropdownstyler), 
                       'value': idx} for idx in range(0,7) if not idx==metadataTypeSelector]
        if metadataSelector1:
            metadataTypeSelectorClearable = False
        else:
            metadataTypeSelectorClearable = True
        return bars, filtered.reset_index()[dispCols].to_dict('records'), metadataSelectorOptions, hist, metadataTypeSelectorOptions1, metadataSelectorOptions1, metadataTypeSelectorClearable
    app.run_server()